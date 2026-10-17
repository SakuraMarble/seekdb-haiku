/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_CG

#include "sql/code_generator/ob_code_generator.h"
#include "sql/code_generator/ob_static_engine_cg.h"
#include "sql/optimizer/ob_log_plan.h"
#include "sql/optimizer/ob_logical_operator.h"
#include "sql/optimizer/ob_log_operator_factory.h"
#include "sql/optimizer/ob_log_table_scan.h"
#include "sql/optimizer/ob_optimizer_context.h"
#include "sql/resolver/expr/ob_raw_expr.h"
#include "sql/ob_sql_define.h"

namespace oceanbase
{
namespace sql
{

// The text retrieval scan (BMW / WAND / DAAT) projects the rows of one scan
// batch into one datum buffer: the iterator always writes from slot 0, so the
// rows of the following batches overwrite the datum slots of the former one
// and the emitted (id, score) pairs get corrupted when more than one batch is
// emitted (this also hits the plain `LIMIT m OFFSET k` plan, where the topn is
// not pushed down and the scan is scanned to feed a top-n sort above it).
// Use a conservative batch size for every plan whose scan is a text retrieval
// scan: whatever the LIMIT value is, the leading rows are then projected in a
// single batch and stay intact. The cap also keeps a not yet resolvable topk
// limit (e.g. an unbound prepared statement) safe.
static const int64_t TEXT_TOPK_FALLBACK_BATCH_SIZE = 1024;

// Get the integer value of a constant expression. A literal limit is converted
// into a parameterized constant (T_QUESTIONMARK) by the parser, so the value is
// read from the parameter store in that case.
static int get_text_topk_const_value(const ObRawExpr &expr,
                                     const common::ParamStore *params,
                                     bool &is_const,
                                     int64_t &value)
{
  int ret = OB_SUCCESS;
  is_const = false;
  value = 0;
  if (!expr.is_const_expr()) {
    // not a constant expression, ignore it
  } else {
    const ObConstRawExpr &const_expr = static_cast<const ObConstRawExpr &>(expr);
    const ObObj &obj = const_expr.get_value();
    const ObObj *val = &obj;
    if (obj.is_unknown()) {
      const int64_t idx = obj.get_unknown();
      if (NULL == params || idx < 0 || idx >= params->count()) {
        LOG_TRACE("text topk limit param is not available at cg time", K(idx));
        val = NULL;
      } else {
        val = &params->at(idx);
      }
    }
    if (NULL != val) {
      if (!val->is_integer_type()) {
        LOG_TRACE("text topk limit is not an integer type", KPC(val));
      } else {
        int64_t int_value = 0;
        if (OB_FAIL(val->get_int(int_value))) {
          LOG_WARN("failed to get integer value of text topk limit", K(ret), KPC(val));
        } else if (int_value > 0) {
          value = int_value;
          is_const = true;
        }
      }
    }
  }
  return ret;
}

static int collect_text_topk_from_tr_info(const ObTextRetrievalInfo &tr_info,
                                          const common::ParamStore *params,
                                          bool &has_topk,
                                          int64_t &max_topk)
{
  int ret = OB_SUCCESS;
  if (NULL == tr_info.topk_limit_expr_ || !tr_info.need_block_max_scan()) {
    // no topk is pushed down into the block max scan
  } else {
    has_topk = true;
    bool is_const = false;
    int64_t value = 0;
    if (OB_FAIL(get_text_topk_const_value(*tr_info.topk_limit_expr_, params, is_const, value))) {
      LOG_WARN("failed to get text topk limit value", K(ret));
    } else if (is_const && value > max_topk) {
      max_topk = value;
    }
  }
  return ret;
}

static int find_text_topk_in_plan(const ObLogicalOperator &op,
                                  const common::ParamStore *params,
                                  bool &has_text_retrieval,
                                  bool &has_topk,
                                  int64_t &max_topk)
{
  int ret = OB_SUCCESS;
  if (log_op_def::LOG_TABLE_SCAN == op.get_type()) {
    const ObLogTableScan &table_scan = static_cast<const ObLogTableScan &>(op);
    if (table_scan.is_text_retrieval_scan() || table_scan.has_es_match() ||
        table_scan.has_merge_fts_index() || table_scan.has_func_lookup()) {
      has_text_retrieval = true;
    }
    if (OB_FAIL(collect_text_topk_from_tr_info(table_scan.get_text_retrieval_info(),
                                               params, has_topk, max_topk))) {
      LOG_WARN("failed to collect text topk from text retrieval info", K(ret));
    } else {
      const ObIArray<ObTextRetrievalInfo> *tr_info_arrays[] = {
          &table_scan.get_match_tr_infos(),
          &table_scan.get_merge_tr_infos(),
          &table_scan.get_lookup_tr_infos(),
          &table_scan.get_vec_iter_tr_infos()};
      const int64_t array_cnt = static_cast<int64_t>(sizeof(tr_info_arrays) / sizeof(tr_info_arrays[0]));
      for (int64_t i = 0; OB_SUCC(ret) && i < array_cnt; ++i) {
        if (tr_info_arrays[i]->count() > 0) {
          has_text_retrieval = true;
        }
        for (int64_t j = 0; OB_SUCC(ret) && j < tr_info_arrays[i]->count(); ++j) {
          if (OB_FAIL(collect_text_topk_from_tr_info(tr_info_arrays[i]->at(j),
                                                     params, has_topk, max_topk))) {
            LOG_WARN("failed to collect text topk from tr info array", K(ret), K(i), K(j));
          }
        }
      }
    }
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < op.get_num_of_child(); ++i) {
    const ObLogicalOperator *child = op.get_child(i);
    if (OB_ISNULL(child)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("unexpected null child in logical plan", K(ret), K(i));
    } else if (OB_FAIL(find_text_topk_in_plan(*child, params, has_text_retrieval, has_topk, max_topk))) {
      LOG_WARN("failed to find text topk in child plan", K(ret));
    }
  }
  return ret;
}

static int adjust_batch_size_for_text_topk(const ObLogPlan &log_plan, int64_t &batch_size)
{
  int ret = OB_SUCCESS;
  const ObLogicalOperator *root = log_plan.get_plan_root();
  if (batch_size <= 0 || NULL == root) {
    // batch processing is disabled, do nothing
  } else {
    bool has_text_retrieval = false;
    bool has_topk = false;
    int64_t max_topk = 0;
    const common::ParamStore *params = log_plan.get_optimizer_context().get_params();
    if (OB_FAIL(find_text_topk_in_plan(*root, params, has_text_retrieval, has_topk, max_topk))) {
      LOG_WARN("failed to find text retrieval topk in plan", K(ret));
    } else if (!has_text_retrieval) {
      // no text retrieval scan in the plan, keep the batch size
    } else {
      // NOTE: the LIMIT/OFFSET values of a text retrieval plan are stored as
      // parameters (topk_limit_expr_), so one cached plan can serve any LIMIT
      // and OFFSET value. The batch size must therefore be large enough for
      // any of them: otherwise the BMW/DAAT iterator overwrites the datum
      // slots of the former batch (it always writes from slot 0) and the
      // emitted (id, score) pairs are corrupted as soon as the scan emits more
      // than one batch (e.g. `LIMIT 10 OFFSET 5` with a batch size of 16, or
      // `LIMIT 20` with a batch size of 16).
      // The batch size is raised up to the fallback cap, so that all the rows
      // needed by any LIMIT (and by the top-n sort of an OFFSET plan) are
      // projected within a single batch.
      int64_t expected = OB_MAX(max_topk, TEXT_TOPK_FALLBACK_BATCH_SIZE);
      if (expected > TEXT_TOPK_FALLBACK_BATCH_SIZE) {
        expected = TEXT_TOPK_FALLBACK_BATCH_SIZE;
      }
      if (expected > batch_size) {
        LOG_TRACE("enlarge batch size for text retrieval scan",
                  K(batch_size), K(expected), K(max_topk), K(has_topk), K(has_text_retrieval));
        batch_size = expected;
      }
    }
  }
  return ret;
}

int ObCodeGenerator::generate(const ObLogPlan &log_plan,
                              ObPhysicalPlan &phy_plan)
{
  int ret = OB_SUCCESS;
  int64_t batch_size = 0;
  const uint64_t cur_cluster_version = CLUSTER_CURRENT_VERSION;
  OZ(detect_batch_size(log_plan, batch_size));
  if (OB_SUCC(ret) && batch_size > 0) {
    log_plan.get_optimizer_context().set_batch_size(batch_size);
    phy_plan.set_batch_size(batch_size);
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(generate_exprs(log_plan, phy_plan, cur_cluster_version))) {
    LOG_WARN("fail to get all raw exprs", K(ret));
  } else if (OB_FAIL(generate_operators(log_plan, phy_plan, cur_cluster_version))) {
    LOG_WARN("fail to generate plan", K(ret));
  }

  return ret;
}
//1. Generate the old execution plan, used for initializing all expression operators
//   and initialize to ObExpr's op_, for mixed running of new and old expressions, will be removed later
//2. Get all expressions that will be used during execution
//3. Generate all physical expressions
int ObCodeGenerator::generate_exprs(const ObLogPlan &log_plan,
                                    ObPhysicalPlan &phy_plan,
                                    const uint64_t cur_cluster_version)
{
  int ret = OB_SUCCESS;
  ObExecContext *exec_ctx = log_plan.get_optimizer_context().get_exec_ctx();
  CK(NULL != exec_ctx && NULL != exec_ctx->get_physical_plan_ctx());
  CK(exec_ctx->get_my_session() != NULL);
  CK(exec_ctx->get_my_session()->use_rich_format() == exec_ctx->get_physical_plan_ctx()->is_rich_format()
     && exec_ctx->get_my_session()->use_rich_format() == phy_plan.get_use_rich_format());
  if (OB_SUCC(ret)) {
    ObStaticEngineExprCG expr_cg(
        phy_plan.get_allocator(),
        log_plan.get_optimizer_context().get_session_info(),
        exec_ctx->get_sql_ctx()->schema_guard_,
        exec_ctx->get_physical_plan_ctx()->get_original_param_cnt(),
        param_store_->count(),
        min_cluster_version_);
    // init ctx for operator cg
    expr_cg.set_batch_size(phy_plan.get_batch_size());
    if (OB_FAIL(expr_cg.generate(log_plan.get_optimizer_context().get_all_exprs(),
                                 phy_plan.get_expr_frame_info()))) {
      LOG_WARN("fail to generate expr", K(ret));
    } else {
      phy_plan.get_next_expr_id() = phy_plan.get_expr_frame_info().need_ctx_cnt_;
    }
  }

  return ret;
}

int ObCodeGenerator::generate_operators(const ObLogPlan &log_plan,
                                        ObPhysicalPlan &phy_plan,
                                        const uint64_t cur_cluster_version)
{
  int ret = OB_SUCCESS;
  ObStaticEngineCG static_engin_cg(min_cluster_version_);
  if (OB_FAIL(static_engin_cg.generate(log_plan, phy_plan))) {
    LOG_WARN("fail to code generate", K(ret));
  }
  return ret;
}

int ObCodeGenerator::detect_batch_size(
    const ObLogPlan &log_plan, int64_t &batch_size)
{
  int ret = OB_SUCCESS;
  bool vectorize = false;
  bool stop_checking = false;
  batch_size = 0;
  ObBasicSessionInfo *session =
      log_plan.get_optimizer_context().get_session_info();
  ObExecContext *exec_ctx = log_plan.get_optimizer_context().get_exec_ctx();
  CK(NULL != exec_ctx && NULL != exec_ctx->get_physical_plan_ctx());
  bool has_registered_vec_op = false;
  if (OB_ISNULL(log_plan.get_plan_root())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected null plan root", K(ret));
  } else if (OB_FAIL(ObStaticEngineCG::exist_registered_vec_op(*log_plan.get_plan_root(), true,
                                                               has_registered_vec_op))) {
    LOG_WARN("check exist registetered vectorized operator failed", K(ret));
  }
  if (OB_FAIL(ret)) {
  } else if (OB_ISNULL(session)) {
    // empty session disable batch processing
  } else {
    
    double scan_cardinality = 0;
    // TODO bin.lb: move to optimizer and more sophisticated rules
    bool rowsets_enabled = true && GCONF._rowsets_enabled;
    // if tenant config is invalid, use 8 as lob_rowsets_max_rows, compatible to origin behavior
    int64_t lob_rowsets_max_rows = true ? GCONF._lob_rowsets_max_rows : 8;
    const ObOptParamHint *opt_params = &log_plan.get_stmt()->get_query_ctx()->get_global_hint().opt_params_;
    if (OB_FAIL(opt_params->get_integer_opt_param(ObOptParamHint::LOB_ROWSETS_MAX_ROWS, lob_rowsets_max_rows))) {
      LOG_WARN("get integer opt param failed", K(ret));
    } else if (OB_FAIL(opt_params->get_bool_opt_param(ObOptParamHint::ROWSETS_ENABLED, rowsets_enabled))) {
      LOG_WARN("fail to check rowsets enabled", K(ret));
    } else if (rowsets_enabled) {
      // TODO bin.lb; check all sub plans
      OZ(ObStaticEngineCG::check_vectorize_supported(vectorize,
                                                     stop_checking,
                                                     scan_cardinality,
                                                     log_plan.get_plan_root()));
    }
    if (OB_FAIL(ret)) {
    } else if (!vectorize) {
      // set max_batch_size = 1
      // if all physical operator is not registered as vec op, disable vectorization
      if (rowsets_enabled && has_registered_vec_op) {
        batch_size = 1;
      } else {
        batch_size = 0;
      }
    } else if (vectorize) {
      ObArenaAllocator alloc;
      ObRawExprUniqueSet flattened_exprs(true);
      OZ(flattened_exprs.flatten_and_add_raw_exprs(log_plan.get_optimizer_context()
                                                           .get_all_exprs()));
      ObStaticEngineExprCG expr_cg(
          alloc,
          log_plan.get_optimizer_context().get_session_info(),
          exec_ctx->get_sql_ctx()->schema_guard_,
          exec_ctx->get_physical_plan_ctx()->get_original_param_cnt(),
          0,
          exec_ctx->get_min_cluster_version());
      int64_t rowsets_max_rows = GCONF._rowsets_max_rows;
      OZ(expr_cg.detect_batch_size(flattened_exprs, batch_size,
                                   rowsets_max_rows,
                                   GCONF._rowsets_target_maxsize,
                                   scan_cardinality,
                                   lob_rowsets_max_rows));
      // overwrite batch size if hint is specified
      OZ(opt_params->get_integer_opt_param(ObOptParamHint::ROWSETS_MAX_ROWS, batch_size));

      int tmp_ret = OB_SUCCESS;
      tmp_ret = OB_E(EventTable::EN_ENABLE_RANDOM_BATCH_SIZE) OB_SUCCESS;
      if (OB_SUCCESS != tmp_ret) {
        static const int64_t min = 1;
        static const int64_t max = rowsets_max_rows * 2;
        batch_size = common::ObRandom::rand(min, max);
      }
      // The block max scan projects all the pushed down topk rows in one batch,
      // make sure the batch size is large enough to hold them.
      OZ(adjust_batch_size_for_text_topk(log_plan, batch_size));
      LOG_TRACE("detect_batch_size", K(vectorize), K(scan_cardinality), K(batch_size),
                K(rowsets_max_rows), K(tmp_ret));
    }
    // TODO qubin.qb: remove the tracelog when rowsets/batch_size is displayed
    // in plan
    LOG_TRACE("detect_batch_size", K(vectorize), K(scan_cardinality), K(batch_size),
              K(rowsets_enabled), K(batch_size), K(has_registered_vec_op));
  }
  return ret;
}

} // end namespace sql
} // end namespace oceanbase

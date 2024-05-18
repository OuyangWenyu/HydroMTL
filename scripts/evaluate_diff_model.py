# more different size model results
import os
import sys

project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from scripts.mtl_results_utils import (
    plot_multi_single_comp_flow_boxes,
    read_multi_single_exps_results,
)

# 7010 is a smaller model
mtl_q_et_small_model_test_exps = ["expstlq0010", "expmtl0030", "expmtl7010"]
# 2030 is a model with different random seed
mtl_q_et_diffrs_model_test_exps = ["expstlq2030", "expmtl2030"]
# et0030 is a model with same random seed with 2030
mtl_et_q_diffrs_model_test_exps = ["expstlet0030", "expmtl2030"]


result_cache_dir = os.path.join(
    definitions.RESULT_DIR,
    "cache",
)
figure_dir = os.path.join(
    definitions.RESULT_DIR,
    "figures",
    "evaluate_more",
)

(
    exps_q_et_results,
    preds_q_lst,
    obss_q_lst,
) = read_multi_single_exps_results(
    mtl_q_et_diffrs_model_test_exps, return_value=True, ensemble=-1
)
cases_exps_legends_together = [
    "STL_Q",
    "MTL_Q",
]
plot_multi_single_comp_flow_boxes(
    exps_q_et_results,
    cases_exps_legends_together=cases_exps_legends_together,
    save_path=os.path.join(
        figure_dir,
        "mtl_test_flow_boxes_compare_with_diff_randseed.png",
    ),
    rotation=45,
)

(
    exps_et_q_results,
    preds_et_lst,
    obss_et_lst,
) = read_multi_single_exps_results(
    mtl_et_q_diffrs_model_test_exps, return_value=True, ensemble=-1, var_idx=1
)
cases_exps_legends_together = [
    "STL_ET",
    "MTL_ET",
]
plot_multi_single_comp_flow_boxes(
    exps_et_q_results,
    cases_exps_legends_together=cases_exps_legends_together,
    save_path=os.path.join(
        figure_dir,
        "mtl_test_flow_boxes_compare_with_diff_randseed.png",
    ),
    rotation=45,
)


(
    exps_q_et_results_diffmodel,
    _,
    preds_q_lst_diffmodel,
    obss_q_lst_diffmodel,
) = read_multi_single_exps_results(
    mtl_q_et_small_model_test_exps, return_value=True, metric="NSE", ensemble=-1
)
cases_exps_legends_together = [
    "STL",
    "MTL",
    "MTL_small",
]
plot_multi_single_comp_flow_boxes(
    exps_q_et_results_diffmodel[:-1],
    cases_exps_legends_together=cases_exps_legends_together,
    save_path=os.path.join(
        figure_dir,
        "mtl_test_flow_boxes_compare_with_smaller_model.png",
    ),
    rotation=45,
)

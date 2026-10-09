import numpy as np
import pytest
from scripts.eval.paper_metrics import score_arrays


def test_cycle_coincidence_does_not_count_as_exact_completion():
    predictions=np.array([[1,0,1,0]])
    targets=predictions.copy();counts=np.array([1,2,3,4])
    stops=np.array([[3,2,-1,1]])
    out=score_arrays(predictions,stops,targets,counts)
    assert out['stopped_answer_correct'].tolist()==[[True,True,False,False]]
    assert out['joint_success'].tolist()==[[False,True,False,False]]
    assert out['correct_letter_wrong_time'].tolist()==[[True,False,False,False]]
    assert out['missing_stop'].tolist()==[[False,False,True,False]]


def test_count_specific_traces_and_headless_metrics():
    p=np.array([[[1,0,1],[0,0,1],[1,1,1]]])
    targets=np.array([[1,0,1]])
    out=score_arrays(p,np.full((1,3),-2),targets,np.array([1,2,3]))
    assert out['nominal_final_correct'].tolist()==[[True,True,True]]
    assert out['complete_trajectory'].tolist()==[[True,False,False]]
    assert out['first_error_loop'].tolist()==[[0,1,2]]
    assert out['exact_stop'] is None
    with pytest.raises(ValueError): score_arrays(p,np.array([[-2,1,1]]),targets,np.arange(1,4))


def test_fixed_prompt_risk_set_censoring_and_recovery():
    from scripts.dataset.pointer import generate_unconditioned_example
    from scripts.eval.paper_metrics import diagnostic_rows
    from scripts.eval.pointer_failure_metrics import reference_trace
    tasks=[generate_unconditioned_example(seed,4,'benchmark',i,'permutation') for i,seed in enumerate((41,43))]
    target=np.array([[ord(c)-65 for c in reference_trace(dict(t.mapping),t.initial_state,4)] for t in tasks])
    p=target.copy();p[0,1]=(p[0,1]+1)%26;p[0,3]=(p[0,3]+1)%26
    metadata=[dict(graph_index=i,example_id=t.example_id,mapping_sha256=t.mapping_sha256,
                   graph_mode='permutation',dataset_seed=307,cycle_period=26,transient_length=0) for i,t in enumerate(tasks)]
    first,loops,episodes=diagnostic_rows(tasks,metadata,p,np.empty(0),np.array([4]),np.array([[4],[4]]),'test',[4])
    assert [r['at_risk'] for r in loops]==[2,2,1,1]
    assert [r['survival'] for r in loops]==[1,.5,.5,.5]
    assert first[0]['loop']==2 and first[0]['recoveries']==1
    assert [(r['start_loop'],r['duration'],r['right_censored']) for r in episodes]==[(2,1,False),(4,1,True)]
    assert first[0]['r_prediction'] is None

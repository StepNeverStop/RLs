from Algorithms.tf2algos.expl_expt import ExplorationExploitationClass


def test_exploration_exploitation_class():
    my_expl = ExplorationExploitationClass(eps_initial=1, eps_final=0.2, eps_final_episode=0.01, eps_evaluation=0,
                                           eps_annealing_episode=50, start_episode=0, max_episode=100)
    assert my_expl.get_esp(0) == 1
    assert my_expl.get_esp(0, evaluation=True) == 0
    assert my_expl.get_esp(80, evaluation=True) == 0
    assert my_expl.get_esp(2) < 1
    assert my_expl.get_esp(50) == 0.2
    assert my_expl.get_esp(51) < 0.2
    assert my_expl.get_esp(100) >= 0.01

    my_expl = ExplorationExploitationClass(eps_initial=0.2, eps_final=0.1, eps_final_episode=0, eps_evaluation=0,
                                           eps_annealing_episode=1000, start_episode=0, max_episode=10000)
    assert my_expl.get_esp(0) == 0.2
    assert my_expl.get_esp(0, evaluation=True) == 0
    assert my_expl.get_esp(500, evaluation=True) == 0
    assert my_expl.get_esp(500) < 0.2
    assert my_expl.get_esp(1000) == 0.1
    assert my_expl.get_esp(2000) < 0.1
    assert my_expl.get_esp(9000) > 0

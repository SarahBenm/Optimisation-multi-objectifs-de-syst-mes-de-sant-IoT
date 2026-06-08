import mo_gymnasium as mo_gym
import numpy as np
import envs
import csv
from src.modules.commun import Constant
from src.modules.Tools import Tools
import time
import os

# from src.scripts.delete_wandb_dir import delete_directory
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import (
    MPMOQLearning,
)
from src.modules.MultiCloud import MultiCloud
from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.performance_indicators import (
    cardinality,
    expected_utility,
    hypervolume,
    igd,
    maximum_utility_loss,
    sparsity,
)
from morl_baselines.common.weights import equally_spaced_weights
from pymoo.util.ref_dirs import get_reference_directions


GAMMA = 1
reward_dim = Constant.number_objectives

# IMPORTANT____________________________________________________________________________________________
# Init Env by giving it the service querry to optimize + the folder where to find the PREPROCESSED data
# number_clouds_list = [5, 10, 20]
number_clouds_list = [20]
"""
    np.array([0, 1, 2]),
    np.array([7, 9, 12]),
    np.array([0, 3, 6]),
    np.array([0, 7, 14]),
    np.array([12, 30, 44]),
    np.array([2, 10, 22]),
    np.array([5, 6, 7]),
    np.array([10, 15, 36]),
    np.array([10, 36, 40]),
    np.array([14, 20, 32]),
    np.array([0, 1, 2, 3, 4]),
    np.array([0, 2, 13, 24, 25]),
    np.array([10, 13, 16, 30, 38]),
    np.array([0, 7, 18, 29, 43]),
    np.array([1, 3, 4, 6, 11]),
    np.array([2, 10, 11, 25, 31]),
    np.array([0, 5, 7, 13, 28]),
    np.array([1, 5, 13, 22, 39]),
    np.array([3, 6, 9, 17, 27]),
    np.array([14, 20, 32, 39, 46]),
    np.array([0, 1, 2, 3, 4, 5, 6, 7]),
    np.array([2, 4, 10, 14, 20, 26, 27, 32]),
    np.array([0, 3, 8, 14, 22, 28, 35, 41]),
    np.array([1, 6, 12, 19, 24, 29, 37, 44]),
    np.array([2, 4, 9, 13, 21, 27, 32, 40]),
    np.array([5, 7, 11, 17, 23, 30, 34, 42]),"""
serviceQuerry_list = [
    np.array([10, 15, 20, 25, 31, 33, 38, 43]),
    np.array([16, 18, 26, 36, 39, 45, 46, 41]),
    np.array([0, 4, 9, 14, 20, 26, 32, 38]),
    np.array([1, 7, 11, 17, 22, 28, 35, 44]),
]

for number_clouds in number_clouds_list:
    MultiCloud_data_dir = f"./src/data/preprocessedData/NC_{number_clouds}_NS_50/NC_{number_clouds}_NS_50_01"
    for serviceQuerry in serviceQuerry_list:
        file_path = f"./results/{number_clouds}clouds/mpmoql.csv"
        """if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)

                # Write header
                writer.writerow(
                    [
                        "composition",
                        "HV",
                        "Sparsity",
                        "Expected Utility",
                        "Card",
                        "IGD",
                        "MUL",
                        "execution_time",
                    ]
                )"""
        # ______________________________________________________________________________________________________

        env = mo_gym.MORecordEpisodeStatistics(
            mo_gym.make(
                "env/SelectService-mpmoql",
                preprocessed_data_dir=MultiCloud_data_dir,
                service_querry=serviceQuerry,
            ),
            gamma=GAMMA,
        )
        eval_env = mo_gym.make(
            "env/SelectService-mpmoql",
            preprocessed_data_dir=MultiCloud_data_dir,
            service_querry=serviceQuerry,
        )

        ref_front = env.unwrapped.pareto_front()

        agent = MPMOQLearning(
            env,
            learning_rate=0.3,  # 0.3
            gamma=GAMMA,
            use_gpi_policy=True,
            dyna=True,
            dyna_updates=5,
            initial_epsilon=1,
            final_epsilon=0.01,
            epsilon_decay_steps=int(2e5),  # int(2e5)
            weight_selection_algo="ols",
            epsilon_ols=0.0,
            project_name=f"MPMOQL_{number_clouds}clouds",
            experiment_name=f"MPMOQL_{Tools.list_to_string(serviceQuerry)}",
            log=False,
            seed=17,
        )
        start_time = time.time()
        current_front = agent.train(
            total_timesteps=10000,
            eval_env=eval_env,
            ref_point=np.full(Constant.number_objectives, -1),
            timesteps_per_iteration=int(1000),
            known_pareto_front=ref_front,
            eval_freq=10,
        )
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        filtered_front = list(filter_pareto_dominated(current_front))
        hv = hypervolume(np.array([-1] * reward_dim), filtered_front)
        sp = sparsity(filtered_front)
        eum = expected_utility(
            filtered_front,
            weights_set=equally_spaced_weights(reward_dim, 50),
        )
        card = cardinality(filtered_front)

        if ref_front is not None:
            generational_distance = igd(
                known_front=ref_front, current_estimate=filtered_front
            )
            mul = maximum_utility_loss(
                front=filtered_front,
                reference_set=ref_front,
                weights_set=get_reference_directions("energy", reward_dim, 50).astype(
                    np.float32
                ),
            )
        else:
            generational_distance = None
            mul = None

        # Write indicators to a CSV
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write data row (you can name your indicators as needed)
            writer.writerow(
                [
                    Tools.list_to_string(serviceQuerry),
                    hv,
                    sp,
                    eum,
                    card,
                    (
                        generational_distance
                        if generational_distance is not None
                        else "N/A"
                    ),
                    mul if mul is not None else "N/A",
                    execution_time,
                ]
            )

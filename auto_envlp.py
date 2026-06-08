import mo_gymnasium as mo_gym
import numpy as np
import envs
import csv
import os
from src.modules.commun import Constant
from src.modules.Tools import Tools
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.performance_indicators import (
    cardinality,
    expected_utility,
    hypervolume,
    igd,
    maximum_utility_loss,
    sparsity,
)
import time
from morl_baselines.common.weights import equally_spaced_weights
from pymoo.util.ref_dirs import get_reference_directions

GAMMA = 1
reward_dim = Constant.number_objectives

# IMPORTANT____________________________________________________________________________________________
# Init Env by giving it the service querry to optimize + the folder where to find the PREPROCESSED data
number_clouds_list = [10]
serviceQuerry_list = [
    np.array([2, 4, 10, 14, 20, 26, 27, 32]),
]

for number_clouds in number_clouds_list:
    MultiCloud_data_dir = f"./src/data/preprocessedData/NC_{number_clouds}_NS_50/NC_{number_clouds}_NS_50_01"

    for serviceQuerry in serviceQuerry_list:
        file_path = f"./results/Execution_results/{number_clouds}clouds/envlp.csv"
        # Write indicators to a CSV
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
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
                )
        # ______________________________________________________________________________________________________

        env = mo_gym.MORecordEpisodeStatistics(
            mo_gym.make(
                "env/SelectService-pcn",
                preprocessed_data_dir=MultiCloud_data_dir,
                service_querry=serviceQuerry,
            ),
            gamma=GAMMA,
        )
        eval_env = mo_gym.make(
            "env/SelectService-pcn",
            preprocessed_data_dir=MultiCloud_data_dir,
            service_querry=serviceQuerry,
        )

        ref_front = env.unwrapped.pareto_front()

        agent = Envelope(
            env,
            max_grad_norm=0.1,
            learning_rate=3e-3,
            gamma=GAMMA,
            batch_size=64,
            net_arch=[256, 256, 256, 256],
            buffer_size=int(2e6),
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=50000,
            initial_homotopy_lambda=0.0,
            final_homotopy_lambda=1.0,
            homotopy_decay_steps=1000,
            learning_starts=100,
            envelope=True,
            gradient_updates=1,
            target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
            tau=1,
            log=False,
            project_name=f"Envelope_{number_clouds}clouds",
            experiment_name=f"Envelope_{Tools.list_to_string(serviceQuerry)}",
        )
        start_time = time.time()
        current_front = agent.train(
            total_timesteps=10000,
            total_episodes=None,
            weight=None,
            eval_env=eval_env,
            ref_point=np.array([-1, -1, -1, -1, -1, -1]),
            known_pareto_front=ref_front,
            num_eval_weights_for_front=100,
            eval_freq=100,
            reset_num_timesteps=False,
            reset_learning_starts=False,
        )
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        print("current_front:", current_front)
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

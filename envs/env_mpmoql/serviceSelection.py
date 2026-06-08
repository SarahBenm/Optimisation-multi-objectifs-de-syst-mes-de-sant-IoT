from gymnasium import Env
import numpy as np
from typing import Optional, List
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from src.modules.MultiCloud import MultiCloud
from src.modules.commun import Constant
from src.modules.Pareto import ParetoFront
from src.modules.Compositon import Composition
from src.modules.Tools import Tools


class SelectService(Env):

    def __init__(self, preprocessed_data_dir: str, service_querry: list[int]):

        self.number_serciceInComps = len(service_querry)
        self.serviceQuerry = service_querry
        self.preprocessed_data_dir = preprocessed_data_dir
        self.multicloud = MultiCloud(
            serviceIds=self.serviceQuerry, multiCloud_dir=self.preprocessed_data_dir
        )

        def get_Step_Actions():
            """
            This function returns a list of the possible step actions in a given multicloud
            The number of actions depends on the number of clouds,
            exemple actions for multicloud with 20 clouds are (6 actions, 5 of them are step actions the last in to go to next service in composition) :
                go to next cloud with a step of 1 , 2 , 4 , 8 , 16 cloud
            """
            power = 1
            actionList = []
            Nb_cloud = self.multicloud.getNumberClouds()
            while power <= Nb_cloud:
                actionList.append(power)
                power *= 2
            return actionList

        self.actions_step_list = get_Step_Actions()
        self.numberActions = (
            len(self.actions_step_list) + 1
        )  # +1 because we have step actions to change cloud indice in a given service + one  action to go to next service in composition
        self.punishment_cnst = -1
        self.numberClouds = self.multicloud.getNumberClouds()

        # Obsercvation space ( States )__________________________________________
        self.numberObs = self.numberClouds * self.number_serciceInComps + 1
        self.observation_space = Discrete(self.numberObs)  # Discrete(self.numberObs)
        """observation_space:
            exemple : service 1 cloud 5 (the second service in cloud 5, we start from service 0): 
                state = 1*self.numberClouds(let's say 20) + 5 =  25 """

        # Action space __________________________________________________
        self.action_space = Discrete(self.numberActions)
        """action_space :
            0 : change the clould of the current service cloud_id +=1 
            2^0 : change the clould of the current service cloud_id +=2
            2^1 : change the clould of the current service cloud_id +=4
            and so on ... 
            last action : Go to next service in composition 
            """

        # Reward space __________________________________________________
        self.reward_dim = Constant.number_objectives
        self.reward_space = Box(
            low=np.full(Constant.number_objectives, self.punishment_cnst),
            high=np.full(Constant.number_objectives, self.number_serciceInComps),
            dtype=np.float64,
        )
        """Exemple : in a Querry of 3  services and 6 objectives 
                The maximum reward we can have is [1,1,1,1,1,1]
                The maximum accumulatve reward we can get is [3,3,3,3,3,3]
                wrong movement we get punishement of [-1,-1,-1,-1,-1,-1]
          """
        # ______________________________________________________________________

        self.reset(seed=17)

    def reset(self, seed=17, options: Optional[dict] = None):
        # Reset observation
        super().reset(seed=seed)

        self.current_service_ind = 0  # first service first cloud
        # Initialize Clouds of services : Give each service in composition the first availible cloud
        self.cloud_for_ser = self.multicloud.init_clouds_for_services()
        self.services = []
        self.composition = None
        info = {}
        return self._get_obs(), info

    def _get_obs(self):
        """
        This function is used to get  encoded state
        this is used because morl-baselines doesn't support dict observation space
        And MPMOQL request the state to be discreat
        """
        if self.current_service_ind < self.number_serciceInComps:
            # if it is not the final state
            stateDict = {
                "service": self.current_service_ind,
                "cloud_id": self.cloud_for_ser[
                    self.current_service_ind
                ].get_cloud_info()["id"],
            }
        else:
            # if it is the final state
            stateDict = {
                "service": self.current_service_ind,
                "cloud_id": 0,
            }
        return self.encode_state(stateDict)

    def encode_state(self, stateDict):
        return np.array(
            [stateDict["service"] * self.numberClouds + stateDict["cloud_id"]]
        )

    def step(self, action):
        terminated = False
        current_serv = self.current_service_ind

        # Initialize the reward vector :
        reward = np.full(Constant.number_objectives, 0)
        # Get current cloud for current service and it id :
        current_cloud = self.cloud_for_ser[current_serv]
        current_cloud_id = current_cloud.get_cloud_info()["id"]
        # Handle actions___________________________________________________________
        if action == self.numberActions - 1:  # go to next state (last action)
            # get your reward
            if (
                current_cloud.get_serviceById(self.serviceQuerry[current_serv])
                is not None
            ):
                reward = current_cloud.get_serviceById(
                    self.serviceQuerry[current_serv]
                ).getrewardVect()
                # Save the selected service in a list so we can make it in the final composition later
                self.services.append(
                    self.multicloud.get_service_by_id(
                        current_cloud_id, self.serviceQuerry[self.current_service_ind]
                    )
                )
            else:
                reward = np.full(Constant.number_objectives, self.punishment_cnst)
                terminated = True

            # Move to next state
            self.current_service_ind += 1
            current_serv += 1

        else:  # We have a step action , we re gonna change the cloud indice of the current service the agent is in
            if (
                current_cloud.get_cloud_info()["id"]  # Movement is not valid :
                >= self.numberClouds - self.actions_step_list[action]
            ):
                reward = np.full(Constant.number_objectives, self.punishment_cnst)
                terminated = True

            elif (  # Service doesn't  exist in the cloud
                current_cloud.service_is_dispo(
                    service_id=self.serviceQuerry[current_serv]
                )
                == False
            ):
                reward = np.full(Constant.number_objectives, self.punishment_cnst)
                terminated = True
            else:  # Valid movement + service exist in cloud :
                self.cloud_for_ser[current_serv] = self.multicloud.get_CloudById(
                    current_cloud_id + self.actions_step_list[action]
                )

        # If Agent reached the end (terminal state):
        if self.current_service_ind == self.number_serciceInComps:
            # 1. Get reward for composition (nb clouds score QoS):
            self.composition = Composition(self.services)
            # 2. reward number of clouds QoS  :
            self.composition.calculate_reward_numberClouds(reward)
            terminated = True

        info = {}
        return self._get_obs(), reward, terminated, False, info

    def pareto_front(self) -> List[np.ndarray[float]]:
        """
        This function returns the true pareto front of the environment found by dominance rule and saved in a csv
        """
        return ParetoFront.calculate_Pareto_front(self.multicloud)

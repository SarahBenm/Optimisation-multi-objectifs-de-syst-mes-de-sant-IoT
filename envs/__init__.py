from gymnasium.envs.registration import register


register(
    id="env/SelectService-mpmoql",
    entry_point="envs.env_mpmoql:SelectService",
    max_episode_steps=300,
    kwargs={"preprocessed_data_dir": None, "service_querry": None},
)


register(
    id="env/SelectService-pcn",
    entry_point="envs.env_pcn:SelectService",
    max_episode_steps=300,
    kwargs={"preprocessed_data_dir": None, "service_querry": None},
)

register(
    id="env/SelectService-dump",
    entry_point="envs.env_dump:SelectService_dump",
    max_episode_steps=300,
    kwargs={"preprocessed_data_dir": None, "service_querry": None},
)

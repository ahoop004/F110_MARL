"""Per-event recovery configuration and controller-state synchronization."""
from collections.abc import Mapping


def validate_respawn(config, agent_ids):
    if config is None:
        return {}
    if not isinstance(config, Mapping) or set(config) - {
        "boundary_agents", "collision_agents", "collision_placement"
    }:
        raise ValueError("respawn accepts boundary_agents, collision_agents and collision_placement")
    for key in ("boundary_agents", "collision_agents"):
        ids = config.get(key, [])
        if (not isinstance(ids, list) or any(not isinstance(a, str) for a in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= set(agent_ids)):
            raise ValueError(f"respawn.{key} must list unique known agent IDs")
    if config.get("collision_placement", "nearest_centerline") != "nearest_centerline":
        raise ValueError("respawn.collision_placement must be nearest_centerline")
    return dict(config)


def reset_respawned(infos, *, controllers, actions, observations):
    """Reset physical command memory, keeping policy weights and reward history."""
    respawned = {aid for aid, info in infos.items() if info.get("respawned")}
    for aid in respawned:
        for components in (controllers, actions, observations):
            component = components.get(aid)
            if component is not None and hasattr(component, "reset"):
                component.reset()
    return respawned

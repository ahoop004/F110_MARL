"""Lap summaries shared by serial episodes and parallel monitoring."""


def episode_lap_summary(info, metrics):
    """Use mean learner laps and sample-weighted valid lap times for teams.

    PPO supplies the last measured lap directly. MAPPO's canonical facts
    supply valid lap samples; finish duration is never used as a lap time.
    """
    race = metrics.get("race_record", {})
    laps = race.get("mean_learner_laps")
    if laps is None:
        laps = info.get("lap_count")
    lap_time = metrics.get("lap_time_s")
    if lap_time is None:
        samples = [(a["mean_valid_lap_time_s"], a.get("valid_lap_count", 0))
                   for a in race.get("agents", {}).values()
                   if a.get("team") == "trainable" and a.get("mean_valid_lap_time_s") is not None]
        count = sum(n for _, n in samples)
        if count:
            lap_time = sum(value * n for value, n in samples) / count
    outcomes = metrics.get("agent_outcomes", {})
    outcome = (" ".join(f"{aid}:{value}" for aid, value in outcomes.items())
               if len(outcomes) > 1 else next(iter(outcomes.values()), info.get("outcome", "?")))
    return laps, lap_time, outcome

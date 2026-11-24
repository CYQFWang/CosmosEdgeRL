import torch
from verl.protocol import DataProto

from training.actor_batch import prepare_actor_batch
from training.reward import Chunk, score_episode, update_values, visual_stage


def main():
    chunks = [Chunk(0, 1, 32, real_stage=1, dream_stage=0), Chunk(1, 0, 17, wrong_grabs=1)]
    scored = score_episode(chunks, {})
    assert [row["reward"] for row in scored] == [1, -2]
    assert scored[0]["honesty"] == -1 and scored[1]["honesty"] is None
    assert abs(scored[0]["advantage"] - (1 - 0.95 * 0.8 * 2)) < 1e-6
    assert abs(update_values([chunks], {}, rate=1)[0] - (1 - 0.95 * 2)) < 1e-6
    assert visual_stage(False, False, 1) == 0
    assert visual_stage(False, True, 1) == 2
    assert score_episode([Chunk(1, 2, 8, success=True)], {2: 100})[0]["advantage"] == 3
    assert score_episode([Chunk(1, 1, 8)], {1: 2})[0]["advantage"] < 0

    batch = DataProto.from_dict(tensors={
        "video_latents_clean": torch.zeros(2, 4, 9, 2, 2),
        "action_latents_clean": torch.zeros(2, 33, 8),
    })
    advantages = [row["advantage"] for row in scored]
    honesty = [row["honesty"] for row in scored]
    args = (batch, advantages, honesty, [32, 17], [1.0, 0.94, 0.83, 0.62])
    base = prepare_actor_batch(*args).batch["reward_prob"].clone()
    prepare_actor_batch(*args, honesty_weight=0.25)
    probs = batch.batch["reward_prob"]
    assert probs.shape == (2, 4, 2)
    assert torch.equal(base[..., 1], probs[..., 1])
    assert (probs[0, :, 0] < base[0, :, 0]).all()
    assert torch.equal(probs[1], base[1])
    assert batch.batch["action_loss_mask"].sum(1).tolist() == [32, 17]
    assert batch.batch["video_loss_mask"].sum(1).tolist() == [8, 4]
    print("PASS: discrete stage, simulator reward, GAE, terminal handling, DataProto routing and masks")


if __name__ == "__main__":
    main()

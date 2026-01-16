# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_rm_scores_sigmoid_applied_before_token_expand():
    """
    Guardrail for a subtle reward scaling bug.

    In `RewardModelWorker.compute_rm_score` we expand a per-sequence scalar score to token-level by
    placing it on the last response token and leaving other positions as 0. Applying sigmoid *after*
    this expansion would turn those zeros into 0.5 (since sigmoid(0)=0.5), inflating any metric that
    sums token-level scores (e.g. ~0.5*max_response_length ≈ 1024 when max_response_length=2048).
    """
    with open("verl/workers/fsdp_workers.py", "r", encoding="utf-8") as f:
        src = f.read()

    assert "torch.sigmoid(-scores)" in src, "Expected compute_rm_score to apply sigmoid on per-sequence `scores`."
    assert (
        "torch.sigmoid(-token_level_scores)" not in src
    ), "Do not apply sigmoid after token-level expansion; it introduces a 0.5 baseline on zero entries."


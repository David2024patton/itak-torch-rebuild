# test_generate.mojo - Lane B4 proof for the generation loop
# and the greedy sampler, over a mock ForwardModel with KNOWN
# outputs.
#
# The mock's rule (fixed here, computable by hand):
#   step(token, position) records the call and returns a
#   hidden row of width 4: [token, position, 0, 0].
#   logits(hidden) returns a width-8 row of -10.0 except one
#   winner at 10.0, where the winner is
#       w = (token + 1 + (position % 2)) % 5
#   read back out of the hidden row, unless the position is
#   the mock's force_pos, where the winner is force_id.
# Every expected sequence below is worked out from that rule
# in the test's comment before it is asserted.

from std.testing import TestSuite, assert_true, assert_equal

from generate.generate import ForwardModel, generate
from generate.sampler import greedy_argmax

comptime MOCK_HIDDEN = 4
comptime MOCK_VOCAB = 8
comptime EOS_ID = 7


struct MockModel(Copyable, ForwardModel, Movable):
    var calls_tok: List[Int]
    var calls_pos: List[Int]
    var force_pos: Int
    var force_id: Int
    var bad_hidden: Bool

    def __init__(out self):
        self.calls_tok = List[Int]()
        self.calls_pos = List[Int]()
        self.force_pos = -1
        self.force_id = -1
        self.bad_hidden = False

    def step(mut self, token_id: Int, position: Int) raises -> List[Float32]:
        self.calls_tok.append(token_id)
        self.calls_pos.append(position)
        var width = MOCK_HIDDEN
        if self.bad_hidden:
            width = MOCK_HIDDEN - 1
        var hidden = List[Float32]()
        hidden.resize(width, Float32(0.0))
        hidden[0] = Float32(token_id)
        hidden[1] = Float32(position)
        return hidden^

    def logits(self, hidden: List[Float32]) raises -> List[Float32]:
        if len(hidden) != MOCK_HIDDEN:
            raise Error(
                "generate_hidden_shape: mock logits expected hidden width "
                + String(MOCK_HIDDEN)
                + ", got "
                + String(len(hidden))
            )
        var token = Int(hidden[0])
        var position = Int(hidden[1])
        var winner = (token + 1 + (position % 2)) % 5
        if position == self.force_pos:
            winner = self.force_id
        var row = List[Float32]()
        row.resize(MOCK_VOCAB, Float32(-10.0))
        row[winner] = Float32(10.0)
        return row^


def assert_ids(got: List[Int], expected: List[Int], what: String) raises:
    assert_equal(len(got), len(expected), what + ": length")
    for i in range(len(expected)):
        assert_equal(got[i], expected[i], what + ": id at " + String(i))


def test_generate_mechanics_calls_and_ids() raises:
    # Prompt [1, 2, 3] at positions 0, 1, 2. By the rule:
    #   hidden(3, 2): w = (3+1+0) % 5 = 4  -> gen 4
    #   step(4, 3):   w = (4+1+1) % 5 = 1  -> gen 1
    #   step(1, 4):   w = (1+1+0) % 5 = 2  -> gen 2
    #   step(2, 5):   w = (2+1+1) % 5 = 4  -> gen 4
    #   step(4, 6):   w = (4+1+0) % 5 = 0  -> gen 0 (5th, stop)
    # Expected ids [4, 1, 2, 4, 0]; step calls are the prefill
    # (1,0) (2,1) (3,2) then feedback (4,3) (1,4) (2,5) (4,6);
    # the 5th generated id is never fed back.
    var model = MockModel()
    var prompt = List[Int]()
    prompt.append(1)
    prompt.append(2)
    prompt.append(3)
    var got = generate(model, prompt, 5, EOS_ID)
    var expected = List[Int]()
    for v in [4, 1, 2, 4, 0]:
        expected.append(v)
    assert_ids(got, expected, "mechanics ids")
    var exp_tok = List[Int]()
    for v in [1, 2, 3, 4, 1, 2, 4]:
        exp_tok.append(v)
    assert_ids(model.calls_tok, exp_tok, "mechanics call tokens")
    var exp_pos = List[Int]()
    for v in [0, 1, 2, 3, 4, 5, 6]:
        exp_pos.append(v)
    assert_ids(model.calls_pos, exp_pos, "mechanics call positions")


def test_generate_eos_stops_and_is_excluded() raises:
    # Same prompt and rule, but the mock forces the winner to
    # EOS_ID (7) at position 4. Picks: hidden(3,2) -> 4,
    # step(4,3) -> 1, step(1,4) -> 7 = EOS. The plain Rust
    # engine loop (engine.rs generate_inner) checks the pick
    # against EOS before counting or emitting it, so the
    # output is [4, 1]: the EOS pick is the 3rd pick, it is
    # excluded, and it is never fed back through step().
    var model = MockModel()
    model.force_pos = 4
    model.force_id = EOS_ID
    var prompt = List[Int]()
    prompt.append(1)
    prompt.append(2)
    prompt.append(3)
    var got = generate(model, prompt, 8, EOS_ID)
    var expected = List[Int]()
    expected.append(4)
    expected.append(1)
    assert_ids(got, expected, "eos ids (eos excluded)")
    var exp_tok = List[Int]()
    for v in [1, 2, 3, 4, 1]:
        exp_tok.append(v)
    assert_ids(model.calls_tok, exp_tok, "eos call tokens")
    var exp_pos = List[Int]()
    for v in [0, 1, 2, 3, 4]:
        exp_pos.append(v)
    assert_ids(model.calls_pos, exp_pos, "eos call positions")


def test_generate_max_tokens_stop() raises:
    # Prompt [5], no EOS reachable (winners are mod 5, EOS is
    # 7). By the rule:
    #   hidden(5, 0): w = (5+1+0) % 5 = 1 -> gen 1
    #   step(1, 1):   w = (1+1+1) % 5 = 3 -> gen 3
    #   step(3, 2):   w = (3+1+0) % 5 = 4 -> gen 4 (3rd, stop)
    var model = MockModel()
    var prompt = List[Int]()
    prompt.append(5)
    var got = generate(model, prompt, 3, EOS_ID)
    var expected = List[Int]()
    for v in [1, 3, 4]:
        expected.append(v)
    assert_ids(got, expected, "max-tokens ids")
    assert_equal(len(model.calls_tok), 3, "max-tokens call count")


def test_generate_hidden_shape_refusal_propagates() raises:
    # The mock's step returns a width-3 hidden row; its
    # logits() is the shape authority and refuses by name.
    # generate() must propagate that named refusal, not
    # swallow it or mint tokens from a bad row.
    var model = MockModel()
    model.bad_hidden = True
    var prompt = List[Int]()
    prompt.append(1)
    prompt.append(2)
    var refused = False
    try:
        _ = generate(model, prompt, 4, EOS_ID)
    except e:
        refused = String(e).find("generate_hidden_shape") != -1
    assert_true(refused, "hidden shape mismatch must refuse by name")


def test_sampler_tie_picks_lowest_index() raises:
    var logits = List[Float32]()
    for v in [Float32(1.0), Float32(3.0), Float32(3.0), Float32(2.0)]:
        logits.append(v)
    assert_equal(greedy_argmax(logits), 1, "tie goes to the lowest index")


def test_sampler_negative_logits() raises:
    var a = List[Float32]()
    a.append(Float32(-5.0))
    a.append(Float32(-9.0))
    assert_equal(greedy_argmax(a), 0, "least negative wins")
    var b = List[Float32]()
    for v in [Float32(-9.0), Float32(-5.0), Float32(-5.0)]:
        b.append(v)
    assert_equal(greedy_argmax(b), 1, "negative tie goes to lowest index")


def test_sampler_empty_refuses_by_name() raises:
    var empty = List[Float32]()
    var refused = False
    try:
        _ = greedy_argmax(empty)
    except e:
        refused = String(e).find("greedy_argmax_empty_logits") != -1
    assert_true(refused, "empty logits must refuse by name")


def test_generate_empty_prompt_refuses() raises:
    var model = MockModel()
    var prompt = List[Int]()
    var refused = False
    try:
        _ = generate(model, prompt, 4, EOS_ID)
    except e:
        refused = String(e).find("generate_empty_prompt") != -1
    assert_true(refused, "empty prompt must refuse by name")
    assert_equal(len(model.calls_tok), 0, "no step calls on refusal")


def test_generate_bad_max_tokens_refuses() raises:
    var model = MockModel()
    var prompt = List[Int]()
    prompt.append(1)
    var refused = False
    try:
        _ = generate(model, prompt, 0, EOS_ID)
    except e:
        refused = String(e).find("generate_bad_max_tokens") != -1
    assert_true(refused, "max_tokens 0 must refuse by name")
    assert_equal(len(model.calls_tok), 0, "no step calls on refusal")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

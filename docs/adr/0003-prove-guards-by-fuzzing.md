# 3. Guards are proven by fuzzing against eager, and each guard is mutation-tested

A missing guard gives silent wrong results on some later call; a too-strict guard only costs a
recompile. So soundness is checked, not assumed: after each workload the cached programs are
fuzzed with sizes that keep and break the relations the trace found, and any accepted draw whose
result differs from eager fails the release gate. Each guard is then removed in turn: the report
says whether its removal produced wrong results, errors inside the program, or nothing in the
draws. The fuzzer's power is itself tested (it must catch a removed branch guard).

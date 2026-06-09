import sys

patch = """
--- a/examples/neural_grammar_discovery.c
+++ b/examples/neural_grammar_discovery.c
@@ -841,6 +841,18 @@ int main(int argc, char **argv) {
                                   has_arg(argc, argv, "--fail-unlinked-roadmap")) != 0) {
             fprintf(stderr, "failed to write paper claim artifacts\\n");
             return 1;
         }
+        if (has_arg(argc, argv, "--evidence-origin-audit")) {
+            int fail_synthetic = has_arg(argc, argv, "--fail-synthetic-required");
+            if (nn_paper_write_evidence_origin_audit(
+                    paper_claims_output_dir == NULL ? "paper/generated"
+                                                    : paper_claims_output_dir,
+                    fail_synthetic) != 0) {
+                fprintf(stderr, "evidence origin audit: synthetic-only required claims detected\\n");
+                if (fail_synthetic) {
+                    return 1;
+                }
+            }
+        }
         return 0;
     }
     if (arc_verified_selfplay_requested) {
"""

with open('examples/neural_grammar_discovery.c') as f:
    content = f.read()

# Find the "return 0" after the paper claims block
old = """                                  has_arg(argc, argv, "--fail-unlinked-roadmap")) != 0) {
            fprintf(stderr, "failed to write paper claim artifacts\\n");
            return 1;
        }
        return 0;"""

new = """                                  has_arg(argc, argv, "--fail-unlinked-roadmap")) != 0) {
            fprintf(stderr, "failed to write paper claim artifacts\\n");
            return 1;
        }
        if (has_arg(argc, argv, "--evidence-origin-audit")) {
            int fail_synthetic = has_arg(argc, argv, "--fail-synthetic-required");
            if (nn_paper_write_evidence_origin_audit(
                    paper_claims_output_dir == NULL ? "paper/generated"
                                                    : paper_claims_output_dir,
                    fail_synthetic) != 0) {
                fprintf(stderr, "evidence origin audit: synthetic-only required claims detected\\n");
                if (fail_synthetic) {
                    return 1;
                }
            }
        }
        return 0;"""

if old in content:
    content = content.replace(old, new)
    with open('examples/neural_grammar_discovery.c', 'w') as f:
        f.write(content)
    print("added evidence-origin-audit flag to main()")
else:
    print("pattern not found")

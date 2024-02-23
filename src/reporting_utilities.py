

"""
Convenience functions that print reports
"""




def print_hypothesis_test_report(pvalue, null_hypothesis=None, conclusion=None, alpha=0.05):
    print(
        "\n".join([line for line in(
            f"Hypothesis Test Report:",
            f"{chr(0x03B1)} = {alpha}",
            f"p-value = {round(pvalue, 5)}",
            f"H{chr(0x2080)}: {null_hypothesis}",
            f"reject H{chr(0x2080)}?: {'yes' if pvalue < alpha else 'no'}",
            f"conclusion: {conclusion}",
            f"\n"
    ) if "None" not in line
    ]))
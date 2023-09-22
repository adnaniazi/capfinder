import pytest

from capfinder.align import align


class TestAlign:
    # Test that the 'align' function correctly aligns two short sequences and returns the aligned query, alignment, and target strings.
    def test_align_short_sequences(self) -> None:
        query_seq = "ACGT"
        target_seq = "AGT"
        pretty_print_alns = False

        expected_query = "ACGT"
        expected_alignment = " /||"
        expected_target = "-AGT"
        expected_alignment_score = 6

        result_query, result_alignment, result_target, result_alignment_score = align(
            query_seq, target_seq, pretty_print_alns
        )

        assert result_query == expected_query
        assert result_alignment == expected_alignment
        assert result_target == expected_target
        assert result_alignment_score == expected_alignment_score

    # Test that the 'align' function correctly aligns two long sequences and returns the aligned query, alignment, and target strings.
    def test_align_long_sequences(self) -> None:
        query_seq = "ACGTACGTACGTACGTACGT"
        target_seq = "ACGTACGTACGTACGTACGT"
        pretty_print_alns = False

        aln_query, aln, aln_target, _ = align(query_seq, target_seq, pretty_print_alns)

        assert aln_query == query_seq
        assert aln_target == target_seq
        assert len(aln) == len(query_seq)

    # Test that the 'align' function correctly aligns two identical sequences
    def test_align_identical_sequences(self) -> None:
        query_seq = "ATCG"
        target_seq = "ATCG"
        pretty_print_alns = False

        expected_result = ("ATCG", "||||", "ATCG", 20)
        assert align(query_seq, target_seq, pretty_print_alns) == expected_result

    # Test that the 'align' function correctly aligns two sequences with a few mismatches
    def test_align_with_mismatches(self) -> None:
        query_seq = "ATCGATCG"
        target_seq = "ATCGATAG"
        pretty_print_alns = False

        expected_aln_query = "ATCGATCG"
        expected_aln = "||||||/|"
        expected_aln_target = "ATCGATAG"
        expected_aln_score = 31

        aln_query, aln, aln_target, aln_score = align(
            query_seq, target_seq, pretty_print_alns
        )

        assert aln_query == expected_aln_query
        assert aln == expected_aln
        assert aln_target == expected_aln_target
        assert aln_score == expected_aln_score

    # Test that the 'align' function correctly aligns two sequences with a few mismatches and gaps
    def test_align_with_mismatches_and_gaps(self) -> None:
        query_seq = "ACCTAG"
        target_seq = "ATCGTACG"
        pretty_print_alns = False

        expected_aln_query = "-----ACCTAG"
        expected_aln = "     ||/   "
        expected_aln_target = "ATCGTACG---"
        expected_aln_score = 6

        aln_query, aln, aln_target, aln_score = align(
            query_seq, target_seq, pretty_print_alns
        )

        assert aln_query == expected_aln_query
        assert aln == expected_aln
        assert aln_target == expected_aln_target
        assert aln_score == expected_aln_score

    @pytest.fixture(autouse=True)
    def capsys(self, capsys):  # type: ignore
        self.capsys = capsys  # type: ignore

    def test_pretty_print_short_alignment(self) -> None:
        query_seq = "ACGT"
        target_seq = "ACCT"
        pretty_print_alns = True

        expected_output = (
            "Alignment score: 11\n\n" "QRY: ACGT\n\n" "ALN: ||/|\n\n" "REF: ACCT\n\n"
        )

        # Capture the standard output using capsys

        align(query_seq, target_seq, pretty_print_alns)

        # Get the captured output
        result, err = self.capsys.readouterr()  # type: ignore

        assert result.replace(" ", "").replace("\n", "") == expected_output.replace(
            " ", ""
        ).replace("\n", "")

    def test_align_with_indels_at_ends_1(self) -> None:
        query_seq = "ATCGTA"
        target_seq = "TACGAT"
        pretty_print_alns = False

        expected_aln_query = "ATCGTA----"
        expected_aln = "    ||    "
        expected_aln_target = "----TACGAT"
        expected_aln_score = 10

        aln_query, aln, aln_target, aln_score = align(
            query_seq, target_seq, pretty_print_alns
        )

        assert aln_query == expected_aln_query
        assert aln == expected_aln
        assert aln_target == expected_aln_target
        assert aln_score == expected_aln_score

    def test_align_with_indels_at_ends_2(self) -> None:
        query_seq = "CGTA"
        target_seq = "ATCGTACG"
        pretty_print_alns = False

        expected_aln_query = "--CGTA--"
        expected_aln = "  ||||  "
        expected_aln_target = "ATCGTACG"
        expected_aln_score = 20

        aln_query, aln, aln_target, aln_score = align(
            query_seq, target_seq, pretty_print_alns
        )

        assert aln_query == expected_aln_query
        assert aln == expected_aln
        assert aln_target == expected_aln_target
        assert aln_score == expected_aln_score


if __name__ == "__main__":
    a = TestAlign()
    a.test_pretty_print_short_alignment()

import filecmp
import os
import tempfile

from capfinder.visualize_alns import calculate_average_quality, visualize_alns


class TestCalculateAverageQuality:
    # Calculate average quality score for a read with all quality scores equal
    def test_all_quality_scores_equal(self) -> None:
        quality_scores = [30, 30, 30, 30, 30]
        result = calculate_average_quality(quality_scores)
        assert result == 30.0

    # Calculate average quality score for a read with minimum possible quality score
    def test_minimum_quality_score(self) -> None:
        quality_scores = [0]
        result = calculate_average_quality(quality_scores)
        assert result == 0.0

    # Visualize alignments for a single FASTQ file
    def test_visualize_single_fastq_file(self) -> None:
        # Create a temporary directory for the output files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Specify the path to the dummy FASTQ file within the temporary directory
            dummy_fastq_filepath = os.path.join(temp_dir, "dummy.fastq")
            # Create an expected dummy ouptput file path
            expected_dummy_output_filepath = os.path.join(
                temp_dir, "expected_dummy.txt"
            )
            # Create an actual dummy output file path
            dummy_output_filepath = os.path.join(temp_dir, "dummy.txt")

            # Create a dummy FASTQ file
            with open(dummy_fastq_filepath, "w") as dummy_fastq:
                dummy_fastq.write("@6245aef5-3525-42c3-8f77-c30615eb6c0b\n")
                dummy_fastq.write(
                    "TCCTCTAATTCCTTATCACCTATTCCTATCTATACTATTATTATCCTACCTACCCTAAACAGGGTCAATGGTCCTTCTTGTCATGACCAACTG\n"
                )
                dummy_fastq.write("+\n")
                dummy_fastq.write(
                    "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
                )
            with open(expected_dummy_output_filepath, "w") as dummy_output:
                dummy_output.write(">6245aef5-3525-42c3-8f77-c30615eb6c0b 40\n")
                dummy_output.write(
                    "TCCTCTAATTCCTTATCACCTATTCCTATCTATACTATTATTATCCTACCTACCCTAAACAGGGTCAATGGTCCTTCTTGTCATGACCAACTG\n\n"
                )
                dummy_output.write("Alignment Score: 50\n")
                dummy_output.write("QRY: TCCTCTAATTC----CTTAT--CACCTATTCCTATCTATA\n")
                dummy_output.write("ALN:           |    |||||  |||| |  ||||||/|| \n")
                dummy_output.write("REF: ----------CCGGACTTATCGCACC-A--CCTATCCAT-\n\n")
                dummy_output.write("QRY: CTATTATTATCCTACCTACCCT--AAACAGGGTCAATGGT\n")
                dummy_output.write("ALN: | ||/|/||///|//////|||  /|||/|||/|      \n")
                dummy_output.write("REF: C-ATCAGTACTGTNNNNNNCCTGGTAACTGGGAC------\n\n")
                dummy_output.write("QRY: CCTTCTTGTCATGACCAACTG\n")
                dummy_output.write("ALN:                      \n")
                dummy_output.write("REF: ---------------------\n\n\n")

            num_processes = 1
            output_folder = temp_dir
            reference = "CCGGACTTATCGCACCACCTATCCATCATCAGTACTGTNNNNNNCCTGGTAACTGGGAC"

            # Call the function under test
            visualize_alns(
                dummy_fastq_filepath, reference, num_processes, output_folder
            )

            # Assert that the two files are identical
            assert filecmp.cmp(dummy_output_filepath, expected_dummy_output_filepath)

    # Visualize alignments for a multipl FASTQ files in a directoryd
    def test_visualize_multiple_fastq_files(self) -> None:
        # Create a temporary directory for the output files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Specify the path to the dummy FASTQ file within the temporary directory
            dummy_fq1_filepath = os.path.join(temp_dir, "dummy_fq1.fastq")
            dummy_fq2_filepath = os.path.join(temp_dir, "dummy_fq2.fastq")

            # Create an expected dummy ouptput file path
            expected_fq1_output_filepath = os.path.join(
                temp_dir, "expected_dummy_fq1.txt"
            )
            expected_fq2_output_filepath = os.path.join(
                temp_dir, "expected_dummy_fq2.txt"
            )

            # Create an actual dummy output file path
            actual_result_fq1_filepath = os.path.join(temp_dir, "dummy_fq1.txt")
            actual_result_fq2_filepath = os.path.join(temp_dir, "dummy_fq2.txt")

            # Create first dummy FASTQ file
            with open(dummy_fq1_filepath, "w") as dfq1:
                dfq1.write("@6245aef5-3525-42c3-8f77-c30615eb6c0b\n")
                dfq1.write(
                    "TCCTCTAATTCCTTATCACCTATTCCTATCTATACTATTATTATCCTACCTACCCTAAACAGGGTCAATGGTCCTTCTTGTCATGACCAACTG\n"
                )
                dfq1.write("+\n")
                dfq1.write(
                    "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
                )
            with open(expected_fq1_output_filepath, "w") as efq1:
                efq1.write(">6245aef5-3525-42c3-8f77-c30615eb6c0b 40\n")
                efq1.write(
                    "TCCTCTAATTCCTTATCACCTATTCCTATCTATACTATTATTATCCTACCTACCCTAAACAGGGTCAATGGTCCTTCTTGTCATGACCAACTG\n\n"
                )
                efq1.write("Alignment Score: 50\n")
                efq1.write("QRY: TCCTCTAATTC----CTTAT--CACCTATTCCTATCTATA\n")
                efq1.write("ALN:           |    |||||  |||| |  ||||||/|| \n")
                efq1.write("REF: ----------CCGGACTTATCGCACC-A--CCTATCCAT-\n\n")
                efq1.write("QRY: CTATTATTATCCTACCTACCCT--AAACAGGGTCAATGGT\n")
                efq1.write("ALN: | ||/|/||///|//////|||  /|||/|||/|      \n")
                efq1.write("REF: C-ATCAGTACTGTNNNNNNCCTGGTAACTGGGAC------\n\n")
                efq1.write("QRY: CCTTCTTGTCATGACCAACTG\n")
                efq1.write("ALN:                      \n")
                efq1.write("REF: ---------------------\n\n\n")

            # Create second dummy FASTQ file
            with open(dummy_fq2_filepath, "w") as dfq2:
                dfq2.write("@5e7c2a9e-c97d-4ee6-9695-5e18aad25c39\n")
                dfq2.write(
                    "CCCTCTATCCTACTCCAACCTCCCTACACATACCTACCTACCACCCTCATCTACACCCCACCCCCTCCC\n"
                )
                dfq2.write("+\n")
                dfq2.write(
                    "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
                )
            with open(expected_fq2_output_filepath, "w") as efq2:
                efq2.write(">5e7c2a9e-c97d-4ee6-9695-5e18aad25c39 40\n")
                efq2.write(
                    "CCCTCTATCCTACTCCAACCTCCCTACACATACCTACCTACCACCCTCATCTACACCCCACCCCCTCCC\n\n"
                )
                efq2.write("Alignment Score: 28\n")
                efq2.write("QRY: CCCTCTATCCTACTCCAACCTCCCTACACAT---ACCTAC\n")
                efq2.write("ALN:                      ||//||//||   ||| ||\n")
                efq2.write("REF: ---------------------CCGGACTTATCGCACC-AC\n\n")
                efq2.write("QRY: CTA-CCACCCTCATC--TACACCCCACCCCCTCCC-----\n")
                efq2.write("ALN: ||| |||   |||||  |||/////////|||///     \n")
                efq2.write("REF: CTATCCA---TCATCAGTACTGTNNNNNNCCTGGTAACTG\n\n")
                efq2.write("QRY: ----\n")
                efq2.write("ALN:     \n")
                efq2.write("REF: GGAC\n\n\n")

            num_processes = 1
            output_folder = temp_dir
            reference = "CCGGACTTATCGCACCACCTATCCATCATCAGTACTGTNNNNNNCCTGGTAACTGGGAC"

            # Call the function under test
            visualize_alns(temp_dir, reference, num_processes, output_folder)

            # Assert that the two files are identical
            assert filecmp.cmp(expected_fq1_output_filepath, actual_result_fq1_filepath)
            assert filecmp.cmp(expected_fq2_output_filepath, actual_result_fq2_filepath)

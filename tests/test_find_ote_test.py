import os
import tempfile

from capfinder.find_ote_test import find_ote_test, make_coordinates


class TestFindOTETest:
    # FIND OTE in a record in a single FASTQ file
    def test_find_ote_in_dir_of_fastq_files(self) -> None:
        # Create a temporary directory for the output files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Specify the path to the dummy FASTQ file within the temporary directory
            fq1_filepath = os.path.join(temp_dir, "fq1.fastq")
            fq2_filepath = os.path.join(temp_dir, "fq2.fastq")

            # Create filepaths for expected outputs
            expected_fq1_output_filepath = os.path.join(
                temp_dir, "ex_fq1_test_ote_search_results.csv"
            )
            expected_fq2_output_filepath = os.path.join(
                temp_dir, "ex_fq2_test_ote_search_results.csv"
            )

            # Create filepaths for actual outputs
            actual_fq1_output_filepath = os.path.join(
                temp_dir, "fq1_test_ote_search_results.csv"
            )
            actual_fq2_output_filepath = os.path.join(
                temp_dir, "fq2_test_ote_search_results.csv"
            )

            # Create the dummy FASTQ file files
            with open(fq1_filepath, "w") as dfq1:
                dfq1.write("@6245aef5-3525-42c3-8f77-c30615eb6c0b\n")
                dfq1.write(
                    "TCCTCTAATTCCTTATCACCTATTCCTATCTATACTATTATTATCCTACCTACCCTAAACAGGGTCAATGGTCCTTCTTGTCATGACCAACTG\n"
                )
                dfq1.write("+\n")
                dfq1.write(
                    "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
                )
            with open(fq2_filepath, "w") as dfq2:
                dfq2.write("@5e7c2a9e-c97d-4ee6-9695-5e18aad25c39\n")
                dfq2.write(
                    "CCCTCTATCCTACTCCAACCTCCCTACACATACCTACCTACCACCCTCATCTACACCCCACCCCCTCCC\n"
                )
                dfq2.write("+\n")
                dfq2.write(
                    "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
                )

            # Create the expected output files
            hdr_str = "read_id,read_type,reason,alignment_score,left_flanking_region_start_fastq_pos,cap_n1_minus_1_read_fastq_pos,right_flanking_region_start_fastq_pos,roi_fasta,fasta_length\n"
            with open(expected_fq1_output_filepath, "w") as efq1:
                efq1.write(hdr_str)
                efq1.write(
                    "6245aef5-3525-42c3-8f77-c30615eb6c0b,good,good_alignment_in_cap-flanking_regions,51,33,38,44,ACTATTATTAT,93\n"
                )

            with open(expected_fq2_output_filepath, "w") as efq2:
                efq2.write(hdr_str)
                efq2.write(
                    "5e7c2a9e-c97d-4ee6-9695-5e18aad25c39,good,good_alignment_in_cap-flanking_regions,37,47,52,58,CATCTACACCC,69\n"
                )

            # Call the function under test
            find_ote_test(
                input_path=temp_dir,
                reference="CCGGACTTATCGCACCACCTATCCATCATCAGTACTGT",
                cap0_pos=38,
                num_processes=1,
                output_folder=temp_dir,
            )
            with open(expected_fq1_output_filepath) as file1:
                content1 = file1.read()
            with open(actual_fq1_output_filepath) as file2:
                content2 = file2.read()
            assert content1 == content2

            with open(expected_fq2_output_filepath) as file1:
                content1 = file1.read()
            with open(actual_fq2_output_filepath) as file2:
                content2 = file2.read()
            assert content1 == content2

    def test_make_coordinates_all_matches(self) -> None:
        aln_str = "|||||||"
        ref_str = "ATCGATG"
        result_list = make_coordinates(aln_str, ref_str)
        expected_list = [0, 1, 2, 3, 4, 5, 6]
        assert result_list == expected_list

    def test_make_coordinates_with_deletions(self) -> None:
        aln_str = "| |||||"
        ref_str = "ATCGATG"
        result_list = make_coordinates(aln_str, ref_str)
        expected_list = [0, 1, 2, 3, 4, 5, 6]
        assert result_list == expected_list

    def test_make_coordinates_with_insertions(self) -> None:
        aln_str = "| |||||"
        ref_str = "A-CGATG"
        result_list = make_coordinates(aln_str, ref_str)
        expected_list = [0, -1, 1, 2, 3, 4, 5]
        assert result_list == expected_list

    def test_make_coordinates_with_gaps_at_5pime_end_in_query(self) -> None:
        aln_str = "  | |||||"
        ref_str = "--A-CGATG"
        result_list = make_coordinates(aln_str, ref_str)
        expected_list = [-1, -1, 0, -1, 1, 2, 3, 4, 5]
        assert result_list == expected_list

    def test_make_coordinates_with_gaps_at_both_ends_of_query(self) -> None:
        aln_str = "  | |||||  "
        ref_str = "--A-CGATG--"
        result_list = make_coordinates(aln_str, ref_str)
        expected_list = [-1, -1, 0, -1, 1, 2, 3, 4, 5, -1, -1]
        assert result_list == expected_list


Capfinder allows you to retrain the model to include additional cap types. This feature is particularly useful for researchers working with novel or less common cap structures that are not included in the pretrained model that is shipped with capfinder.

### How was the data for current pretrained model generated?

The classifier has currently been trained on 4 different cap classes:

1. Cap 0
2. Cap 1
3. Cap 2
4. Cap 2-1

### Synthesized Oligos for Training Data

Data for these caps was generated by sequencing the following synthesized oligos:

| Cap Type | Oligo Sequence (5' → 3') |
|----------|--------------------------|
| Cap 0    | `GCUUUCGUUCGUCUCCGGACUUAUCGCACCACCUAUCCAUCAUCAGUACUGUN1N2N3N4N5N6CGAUGUAACUGGGACAUGGUGAGCAAUCAGGGAAAAAAAAAAAAAAA` |
| Cap 1    | `GCUUUCGUUCGUCUCCGGACUUAUCGCACCACCUAUCCAUCAUCAGUACUGUmN1N2N3N4N5N6CGAUGUAACUGGGACAUGGUGAGCAAUCAGGGAAAAAAAAAAAAAAA` |
| Cap 2    | `GCUUUCGUUCGUCUCCGGACUUAUCGCACCACCUAUCCAUCAUCAGUACUGUmN1mN2N3N4N5N6CGAUGUAACUGGGACAUGGUGAGCAAUCAGGGAAAAAAAAAAAAAAA` |
| Cap 2-1  | `GCUUUCGUUCGUCUCCGGACUUAUCGCACCACCUAUCCAUCAUCAGUACUGUN1mN2N3N4N5N6CGAUGUAACUGGGACAUGGUGAGCAAUCAGGGAAAAAAAAAAAAAAA` |

Where:

- `N` represents any of the four RNA bases (A, U, C, G)
- `m` preceding a base indicates 2'-O methylation

### Key Considerations for Custom Oligo Design

When designing custom oligos for new cap types, keep in mind:

1. **OTE Sequence Requirement**

   - The OTE sequence `5'-GCUUUCGUUCGUCUCCGGACUUAUCGCACCACCUAUCCAUCAUCAGUACUGU-3'` must be present to the left of the cap.

2. **Sequence After NNNs**

    - The sequence following the NNNs part (`5'-CGAUGUAACUGGGACAUGGUGAGCAAUCAGGGAAAAAAAAAAAAAAA-3'`) can be customized.
    - Recommendations:
        - Include a synthesized polyA tail
        - Keep polyA tail length ≤ 15 bases
        - Ensure total construct length ≥ 105nt for adequate basecall sequencing quality

3. **Cap and N Bases**

    - The cap base(s) and following `N` bases should total 6.
    - This ensures cap bases exist in all possible 4/5-mer contexts to the right.
    - The context to the left of the cap is fixed due to the OTE.

## Example: Oligo Sequence for a New Cap Type (Cap0-m6A)

Let us say you want to extend the classifier to account for a fifth cap - Cap0-m6A. For this cap, the synthesized oligo should resemble the following if we take into consideration the requirements we have explained above:


```
5'-GCUUUCGUUCGUCUCCGGACUUAUCGCACCACCUAUCCAUCAUCAGUACUGUm6A1N2N3N4N5N6CGAUGUAACUGGGACAUGGUGAGCAAUCAGGGAAAAAAAAAAAAAAA-3'
```

Note: `N1` is replaced with `m6A1`.

Next, we will extend cap mappings capfinder uses to accomodate the new cap type.
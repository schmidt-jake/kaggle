# Exploratory Data Analysis (EDA)

## Questions

- How many images per patient?
- How are patient images stratified / distributed across centers?
- Any chance that `image_id` is correlated with the label?
- Is there any overlap b/w train and test in terms of patient ID / center ID?
- Are the source images all the same pixel resolution?
- How can we correct for different image colors?

## Notes

- Classes are not balanced. `CE` is ~72% of the training data, and `LAA` is ~27%.
- Many images seem to have multiple tissue slices (z levels) of the same clot.

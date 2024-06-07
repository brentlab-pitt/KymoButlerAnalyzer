# KymoButlerAnalyzer
 Compile, extract, and analyze data from KymoButler results ouput


# KymoButlerAnalyzer

The following script is intended to batch compile, extract, and analyze data from KymoButler results files. KymoButler is an AI program that analyzes kymographs made by Max Jakobs: https://github.com/MaxJakobs/KymoButler. 

After running KymoButler to analyze kymographs, you should have a results file for each kymograph analyzed. The following script assumes you have a directory with these excel files for each condition of data you have (ex conditions: wildtype vs mutant). This script can either analyze one replicate of an experiment, or multiple replicates at once. It will loop through each set of KymoButler results files, extract different metrics based on direction (ex. anterograde track durations) as well as counting the number of tracks per direction, then compile these results from every file together and save these compiled results in kymo_results folder ({condition_name}_compiled_results.xlsx). Then it will calculate the mean value of each metric per file, and compile that onto a new excel file with the means from the other conditions as well (mean_compiled_results.xlsx). These means can optionally be normalized to the first condition's mean (conditionX*(conditionX_mean/condition1_mean)). Then it performs statistical analysis (t-tests and anova) on the means of every metric, and exports that data into an excel file (statistical_test_results.xlsx). Then it creates violin plots and histogram plots for each metric, and saves these as both PNGs and PDFs. Optionally, it will also create a grid of all kymographs per condition.

### Directions:

1. Create kymographs using Fiji/ImageJ or some other image analysis tool. Save them as PNGs in individual folders in a directory (have different directories for different conditions)
2. Run KymoButler batch script on the directory, it will loop through each folder, do analysis on the PNG in that folder (only have one PNG per folder), then output the results files in the chosen directory
3. Run the following script, it will as you several prompts:

    a. Enter the number of conditions you intend to analyze (statistical analysis necessitates at least 2 conditions)

    b. Enter the name of each condition (this should be the SAME names as your folder names for each condition, make sure these folder names are the same across all replicates)

    c. Enter 'Y' if you intend to analyze more than one replicate of data, 'N' if not

    d. Enter 'Y' if you want to normalize the data to the mean of the first condition, 'N' if not

    e. Enter 'Y' if you want to create a grid of all your kymographs per condition, 'N' if not (note: kymograph PNGs are expected to be within individual folders within the condition folder)
        
        Directory Example: C:\User\username\Desktop\Replicates\Replicate1\Condition1\Kymograph1\Kymograph.png
    
    f. Choose the directory you wish to analyze, if you said 'Y' to replicates, choose the directory that contains all your replicate folders (a replicate is one iteration of an experiment, it is expected that each replicate has the same condition folder names within it), if you wish to analyze a single replicate, choose the directory of one replicate

        Multiple Replicates Chosen Directory: C:\User\username\Desktop\Replicates

        Single Replicate Chosen Directory: C:\User\username\Desktop\Replicates\Replicate1


        KymoButler Results File Directory Example: C:\User\username\Desktop\Replicates\Replicate1\Condition1\kymograph1.xlsx

### Tips for Troubleshooting
1. When analyzing multiple replicates, the chosen directory should only contain the replicate folders, which each contain condition subfolders (make sure the condition folders have the same names across replicates and that you enter them correctly when prompted)
2. Make sure your kymographs are each in their own folder within the condition folders and that there is only one PNG per folder
3. Make sure your KymoButler results files are all located within the condition folders



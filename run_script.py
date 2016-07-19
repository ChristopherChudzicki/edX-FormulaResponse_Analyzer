import FormulaResponseAnalyzer as fra

##################################################
#       Run Stuff
##################################################  
problem_checks_augmented_filename = "MITx_FakeCourse_problem_checks_augmented.csv"
problem_id_front = "i4x-MITx-FakeCourse-problem-"

fra.split_csv_by_problem_id(problem_checks_augmented_filename, problem_id_front)
fra.eval_and_summarize()
# FormulaResponse_Analyzer.py
Group symbolic submissions by numerical equivalence.

![output of FormulaResponse_Analyzer for one problem](https://github.com/ChristopherChudzicki/edX-FormulaResponse_Analyzer/blob/master/problem_output_screenshot.png "Problem Output")


## Background
It is possible within within edX to give wrong-answer feedback on formularesponse ("symbolic input") problems using the hintgroup tag. However, existing edX analytics tools (Insights and the MITx dashboard) group formularesponse submissions by string equivalence rather than mathematical equivalence (e.g., x+x and 2x are grouped separately) which makes identifying common answers somewhat difficult. 

Additionally, formularesponse problems sometimes have grading issues: the edX formularesponse grader checks for equivalence to the instructor provided answer by numerical sampling and instructors must set the sample range manually for each problem. This can lead to several issues:

* Samples may be outside the domain for mathematical functions used in correct submissions. This can cause all correct answers to me marked incorrect, or can cause identical strings to be graded differently for different users. Because edX uses different random samples for each user, if the sampe range for x is x\in[0,2] then the string sqrt(1-x) may be marked as correct for some users but incorrect for others.
* wrong answers may be erroneously marked correct (e.g., x and sin(x) will be graded the same if the x samples are small)
* correct answers may be marked wrong (e.g., in a physics problem cos(x) and sqrt(1-sin(x)) may be the same in a physics problem because x is in the first quadrant, but if grading samples are taken from all quadrants, these answers will be treated differently )

## Purpose
`FormulaReponse_Analyzer.py` is a python script intended to help instructors 

* Group symbolic submissions by (approximate) mathematical equivalence for a single formularesponse problem;
* Validate edX grading for a single formularesponse problem;
* Provide a user interface that enables instructors to quickly identify which formularesponse problems in a given course would benefit from wrong-answer feedback and which problems may have grading issues.

`FormulaReponse_Analyzer.py` also groups submissions to numeric problems, but randomized problems are not properly supported. 

## Functions in FormulaResponse_Analyzer.py
The input to the FormulaResponse_Analyzer.py process is a single, tab-separated `csv` file containing submission information for users and problems within a course. The following columns must be included:

`hashed_username, submission, correctness, problem_id`

FormulaResponse_Analyzer.py does not currently use the problem XML (using problem XML may be desirable in the future, but seems technically difficult). 

FormulaReponse_Analyzer.py has five steps, each implemented as a python function. Intermediate information is written to a new `csv` at the end of each step. Below are some overall comments on these functions. For details about parameters, inputs, and outputs, see the extensive code comments.

1. `split_csv_by_problem_id`:
 * splits single original tab-delimited `csv` file with all users and all problems into into one file per problem
 * Some problems are skipped based on input types. We're mainly interested in symbolic problems, so we skip all problems except those whose input type is `textline`and `formulaequationinput`. The resulting collection of problems includes many non-symbolic problems.(numerical and customresponse problems often use `textline`).
2. `make_eval_csv`:
 * numerically evaluates the submissions to a single problem a specified number of times. Evaluation is performed using edX calc.py library
 * no information is read from problem XML. We detect variables and assign random samples within the range [0,1].
 * `calc.py` is patched to use `numpy.lib.scimath.sqrt` instead of `numpy.sqrt` to avoid sampling errors. [`numpy.sqrt(-4) = NaN`; `numpy.lib.scimath.sqrt(-4)=2j`]. A few other similar functions (e.g., `arcsin` and `power`) are also patched.
 * Evaluations can be complex (e.g., `arcsin(5)` ).
 * By default, evaluations are case-insensitive. (This is the edX default).
 * exact duplicate submissions by the same user are skipped by default.
 * data is handled using the pandas python module 
3. `make_summary_csv`:
 * group problems by equivalent evaluations. “Equivalent” means numerically identical on all evaluations after rounding to a specified number of significant digits.
 * This grouping procedure is obvious not perfect, but seems to work very well within 8MechCx and is very easy to implement using the built-in groupby method in pandas.  
 * for each group, calculate various statistics including 
 * frequency of equivalent (by evaluation) submissions
 * frequency of identical (by string) submissions
 * fraction of submissions we believe are equivalent that edX graded as correct [should be 0 or 1]
 * fraction of identical submissions that edX graded as correct [should be 0 or 1]
4. `make_gui_problem_table`:
 * summarize submissions for a single problem. This is an HTML file built with  the ElementTree python module [plus some javascript]
5. `make_gui_index`:
 * This is an HTML file built with  the `lxml.etree` python module 
 * index of problem summaries for every formularesponse problem in the course
 * In order to make finding “interesting” problems easy, every formularesponse problem has three statistics:
    * Feedback Score (range, 0–1): The wrong answers form a probability distribution. This is just the second moment of that distribution, errorsperror2. This second moment is larger if the wrong answers are of just a few common types. This sort of problem would benefit more from wrong-answer feedback
    * n_correct : the number of submission groups wherein some but not all submissions were marked as correct. Should be 1. If bigger than 1, could indicate grader is not strict enough.
    * n_partial: number of submission groups wherein some but not all submissions were marked as correct. Should be 0. If not zero, could indicate a variety of grading issues.
Index is sortable (using `tablesorter.js`)


## Problem XML
Currently FormulaResponse_Analyzer does not get any information from the problem XML for two reasons:

1. we believe problem XML would be difficult (certainly not impossible) to parse and match to tracking logs. [e.g., XML names and tracking logs names don’t always match] 
2. although problem XML is available to course authors, we do not believe it is available to analytics dashboards

Not using information from the problem XML presents several changes including:

 * we don’t know what variables are expected (this is fairly easy to solve)
 * we do not know if the problem was graded by edX as case sensitive or case insenstive (edX default is case insensitive)

This is not a big problem, as long as the course authors are consistent
we do not know the numerical sampling range intended by the author. [Using a different sample range than the edX grader can be advantageous, as our analysis can be used to validate edX grading and detect problems with inappropriate author set sample ranges]

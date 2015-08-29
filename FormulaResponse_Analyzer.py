# Version 0.5
##################################################
#       Load Stuff
##################################################  
import calc #see https://github.com/edx/edx-platform/tree/master/common/lib/calc/calc
import random #generate random numbers
import pandas #good for storing data. Reproduces some R-like functionality
import numpy  #vectorized numerical math; c
import os
from numbers import Number

#Eceptions:
from pyparsing import ParseException
from calc import UndefinedVariable

##################################################
#       Patch calc.py to be a bit more foregiving about function domains
##################################################  
calc.DEFAULT_FUNCTIONS['sqrt'] = numpy.lib.scimath.sqrt #numpy.sqrt(-4) = nan, numpy.sqrt(-4+0j)=2j; numpy.lib.scimath.sqrt(-4)=2j
calc.DEFAULT_FUNCTIONS['arccos'] = numpy.lib.scimath.arccos
calc.DEFAULT_FUNCTIONS['arcsin'] = numpy.lib.scimath.arcsin
#my_eval_power is the same as calc.eval_power, except it uses numpy.lib.scimath.power(b,a) instead of b**a. That way (-4)^0.3 will not throw errors.
def my_eval_power(parse_result):
    """
    Take a list of numbers and exponentiate them, right to left.

    e.g. [ 2, 3, 2 ] -> 2^3^2 = 2^(3^2) -> 512
    (not to be interpreted (2^3)^2 = 64)
    """
    # `reduce` will go from left to right; reverse the list.
    parse_result = reversed(
        [k for k in parse_result
         if isinstance(k, calc.numbers.Number)]  # Ignore the '^' marks.
    )
    # Having reversed it, raise `b` to the power of `a`.
    power = reduce(lambda a, b: numpy.lib.scimath.power(b,a), parse_result)
    return power
calc.eval_power = my_eval_power


##################################################
#       CSV Stuff
##################################################  
def ensure_dir(path):
    '''Check if path's directory exists, create if necessary. From http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary'''
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

def split_csv_by_problem_id(input_csv_path,problem_id_front,sep='\t',acceptable_response_types=['formularesponse']):
    '''
    (single csv)->many csv
    `input_csv` should contain problem_check table for every problem in the course. This splits the csv into many new csv, one per problem.
    
    input_csv:
    time,   hashed_username,    problem_id,     correctness,    submission,     attempt_number,     input_type,     response_type
    
    output_csv: (one per unique problem_id)
    time,   hashed_username,   submission,     correctness,    attempt_number,     input_type,  response_type
    
    csv are tab-separated by default
    
    '''
    df = pandas.DataFrame.from_csv(input_csv_path,index_col=False,sep=sep)
    print(df.head(10))
    grouped = df.groupby("problem_id")
    for group_name, group_df in grouped:
        problem_name = group_name.replace(problem_id_front,"")
        output_csv_path = "problem_check/" + problem_name + ".csv"
        response_type = group_df['response_type'].iloc[0]
        if response_type in acceptable_response_types:
            ensure_dir(output_csv_path)
            group_df.to_csv(output_csv_path,sep="\t")
        else:
            print("Skipping problem" + problem_name + ", it's response_type is " + response_type)
    return

def make_eval_csv(input_csv_path,output_csv_path=None,n_evals=2,skip_duplicates=True,case_sensitive=False):
    '''
    (csv) -> csv
    input_csv_path: problem_check/<problem_id>.csv
    output_csv_path: if not specified, problem_evaluated/<problem_id>_evaluated.csv
    
    Input csv should at least have columns:
    hashed_username,   submission,     correctness,  response_type
    str                str             boolean       str    
    
    Output has same columns, plus additional columns:
    eval.0,     eval.1,   ...       eval.(n_eval-1)
    float       float               float
    
    evals are numerical evaluations of the submission string. Variables are mapped to floats by the vars_dict_list dictionaries: 
    0. Initialize vars_dict_list =[{},...,{}]
    1. Try to evaluate submission
        if success, move to next submission
        if failure, update vars_dict_list and try to re-evaluate
    
    EXCEPTIONS:
    If a submission cannot be parsed, ParseException is raised
    '''
    #########################
    #Import the csv
    #########################
    df = pandas.DataFrame.from_csv(input_csv_path,index_col=False,sep="\t")
    
    #Remove all whitespace
    df['submission'] = df['submission'].replace(r"\s+",'',regex=True)
    #Remove anything of the form `[...]`, for example `[tab]`
    df['submission'] = df['submission'].replace(r'\[.*\]', '',regex=True)
    #Replace empty submissions by nan
    df['submission'] = df['submission'].replace('',numpy.nan)
    
    if skip_duplicates:
        # If Bob submits "a+b", "a+b", and "b+a", we drop the second "a+b" but keep "b+a"
        df = df.drop_duplicates(subset=["hashed_username","submission"])   
    
    #########################
    #Setup vars_dict_list and updating function
    #########################
    vars_dict_list = [{} for j in range(0,n_evals)] #Initialize vars_dict_list
    funcs_dict = {} #Empty now...edX evaluator requires a vars_dict and a functions_dict. I do not think funcs_dict is EVER used within the edX platform, but it is required.
    
    def update_vars_dict_list(submission,vars_dict_list):
        '''
        Detects variables used in submission and updates vars_dict_list accordingly.
        '''
        try:
            #Extract new variables from submission
            submission = calc.ParseAugmenter(submission)
            submission.parse_algebra()
            #find out which variables are new
            old_vars = set(vars_dict_list[0].keys())
            full_vars = submission.variables_used
            if not case_sensitive:
                old_vars = set([var.lower() for var in old_vars])
                full_vars = set([var.lower() for var in full_vars])
            new_vars = full_vars - old_vars
            #Assign values to each new variable
            for var in list(new_vars):
                for vars_dict in vars_dict_list:
                    vars_dict[var] = random.uniform(0.01,1)
            return vars_dict_list
        except ParseException:
            print("Failed to update varDict")
            return vars_dict_list
            
    
    #########################
    #Construct eval columns
    #########################
    for index, row in df.iterrows():
        #If submission non-empty, then try to evaluate submission using existing vars_dict_list; if an UndefinedVariable exception occurs, attempt to update vars_dict_list
        if pandas.isnull(row['submission']): #edX calc parsers fail on empty submissions, so treat them separately. Replace with infinity
            for sample_index, vars_dict in enumerate(vars_dict_list):
                df.loc[index,'eval.'+str(sample_index) ] = numpy.inf
        elif isinstance(row['submission'], Number): #edX calc parser fails if the submission's python datatype is numeric. If at least one submission is a string, then numbers are converted to strings on import and this is not a problem. Occasionally, all submissions are numeric, and then the python data type is numeric. Treat this case separately.
            for sample_index, vars_dict in enumerate(vars_dict_list):
                df.loc[index,'eval.'+str(sample_index) ] = row['submission']
        else: 
            try:
                for sample_index, vars_dict in enumerate(vars_dict_list):
                    df.loc[index,'eval.'+str(sample_index) ] = calc.evaluator(vars_dict,funcs_dict,row['submission'],case_sensitive=case_sensitive)
            except UndefinedVariable as undefined_variable:
                print("attempting to update vars_dict_list to include: " + str(undefined_variable))
                vars_dict_list = update_vars_dict_list(row['submission'],vars_dict_list)
                for sample_index, vars_dict in enumerate(vars_dict_list):
                    df.loc[index,'eval.'+str(sample_index) ] = calc.evaluator(vars_dict,funcs_dict,row['submission'],case_sensitive=case_sensitive)
                    #If vars_dict fails to update, the submission was invalid edX string.
            except ParseException:
                print("Invalid submissions detected: \n" +"#"*25+ "\n" + str(row)) + "\n"+"#"*25
                raise
    
    #########################
    #Export the csv
    #########################
    if output_csv_path is None: 
        output_csv_path = "problem_evaluated/" + os.path.basename(input_csv_path)[:-4] + "_evaluated" + ".csv"
    #Check if output directory exists. If it doesn't, create it. 
    ensure_dir(output_csv_path)
    
    df.to_csv(output_csv_path,sep="\t")
    
    return

def sig_round(z,n_sig=3,chop_past=12):
    z_copy = numpy.copy(z)
    z_copy = numpy.round(z_copy,decimals=chop_past)
    def round_part(x):
        x.flags.writeable=True
        power = 10.0**(-numpy.floor(numpy.log10(abs(x[numpy.flatnonzero(x)]))) + n_sig - 1)
        x[numpy.flatnonzero(x)] = numpy.round(x[numpy.flatnonzero(x)]*power,0)/power
        return x
    return round_part(z_copy.real) + round_part(z_copy.imag)*1j        

def make_summary_csv(input_csv_path,output_csv_path=None,n_sig=2):
    '''
    input_csv_path: a csv from make_eval_csv, probably something like "problem_evaluated/<problem_id>_evaluated.csv"
    output_csv_path: if not specified, "problem_summary/<problem_id>_evaluated_summary.csv"
    INPUT csv should at least have columns:
    submission,  correctness,  eval.0,  ...,  eval.(n_eval-1), response_type
    string       0/1           float         float              string
    
    OUTPUT csv has statistics on these columns, plus response_type and statistics of two types: 
        eval: aggregated over submissions with numerically equivalent evaluations (to n_sig sig figs)
        subm: aggregated over submissions with identical submission strings
    subm_correctness: of identical submissions, what fraction were marked correct? [should be 0 or 1, might not be because of edX grader random samples]
    subm_count: number of submissions with identical submission strings 
    eval_correctness: of submissions with numerically equivalent evals, what fraction were marked correct? [should be 0 or 1, might not be because of edX grader random samples]
    eval_count: number of submissions with equivalent evaluations
    subm_frequency: this string's frequency among numerically equivalent submissions
    eval_frequency: fraction of total submissions with this numerical evaluation
    '''
    #########################
    #Import the csv and round evals
    #########################
    df = pandas.DataFrame.from_csv(input_csv_path,sep="\t")
    eval_cols = [val for val in df.columns.values if val.startswith('eval')]
    for eval_col in eval_cols:
        #Pandas does not import complex numbers properly from csv. This is a reported issue: https://github.com/pydata/pandas/issues/9379
        if df[eval_col].dtype==object: df[eval_col] = df[eval_col].apply(complex)
        df[eval_col] = sig_round(df[eval_col],n_sig=n_sig)
        #Convert complex numbers to strings so grouby can sort them
        df[eval_col] = df[eval_col].apply(str)

        
    #########################
    #Group and Summarize
    #########################
    #groupby drops nan rows by default. Switch nans to 'empty'
    df['submission'] = df['submission'].replace(numpy.nan,'empty')
    #group by evals and submissions
    subm_grouped = df.groupby(by=eval_cols+['submission',])    
    summary = subm_grouped.aggregate({
                 'correctness':numpy.mean,
                },sort=False)
    #correctness is averaged over submission groups. Rename and convert to float.
    summary.rename(columns={'correctness': 'subm_correctness'}, inplace=True)
    summary['subm_correctness'] = summary['subm_correctness']*1.0
    
    #Size of submission groups
    summary['subm_count'] = subm_grouped.size()
    #Size of everyone
    total = sum(summary['subm_count'])
    #Size of eval groups
    eval_groups = summary.groupby(level=eval_cols).groups.keys()
    for eval_group in eval_groups:
        summary.loc[eval_group,'eval_count']=sum(summary.loc[eval_group,'subm_count'])
        #percent of submissions that **I have declared** numerically equivalent that were graded correct by edX
        summary.loc[eval_group,'eval_correctness']=numpy.mean(summary.loc[eval_group,'subm_correctness'])
    #eval group frequencies: percent of ALL submissions that eval'd numerically equiv
    summary['eval_frequency'] = summary['eval_count']/total
    #submission group frequencies: percent of numerically equivalent submssions that are same string
    summary['subm_frequency']  = summary['subm_count']/summary['eval_count']
    
    summary['response_type'] = df['response_type'].iloc[0]

    summary.sort(columns=['eval_count','subm_count'],inplace=True,ascending=False)

    #########################
    #Export the csv
    #########################
    if output_csv_path is None:
        output_csv_path = "problem_summary/" + os.path.basename(input_csv_path)[:-4] + "_summary" + ".csv"
    
    #Check if output directory exists. If it doesn't, create it. 
    ensure_dir(output_csv_path)
    
    summary.to_csv(output_csv_path,sep="\t")
    
    return

##################################################
#       HTML GUI Stuff
##################################################  
import lxml.etree as ET
import copy
def make_gui_problem_table(input_csv_path,output_html_path=None):
    '''
    input_csv_path: a csv from make_summary_csv, probably something like "problem_summary/<problem_id>_evaluated_summary.csv"
    '''
    #########################
    #Import CSV
    #########################
    df = pandas.DataFrame.from_csv(input_csv_path,sep="\t")
    #restore multi-index
    eval_cols = ['eval.0']+[val for val in df.columns.values if val.startswith('eval.')]
    df = df.set_index(keys = eval_cols[1:], append=True)
    #group by evals, get group names, and sort by eval_count
    grouped = df.groupby(level=eval_cols,sort=False)
    groups = [name for name, group in grouped]
    groups = sorted(groups, key=lambda name: grouped.get_group(name)['eval_count'][0], reverse=True) 

    #########################
    #Import HTML Template
    #########################
    template_path = "gui/templates/gui_problem_template.html"
    with open(template_path,'r') as f:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(f,parser)
    root = tree.getroot()
    #Get tbody from template. This is where data goes.
    table = list(tree.iterfind('body/table'))[0]
    template_tbody = list(table.iterfind('tbody'))[0]
    template_subm_tr = list(table.iterfind('tbody/tr[2]/td/div/table/tbody/tr'))[0]
    table.remove(template_tbody)
    
    def populate_tbody(group,tbody_elem,eval_group_columns,subm_group_columns,max_subm_rows=20):
        #Add main (eval) row, with statistics for all submissions with equivalent numerical evaluations
        for indx, elem in enumerate(tbody_elem.iterfind('tr[1]/td')):
            elem.text = str( grouped.get_group(group)[ eval_group_cols[indx] ][0] )
        #Add detail (subm) rows, with statistics for each submission string in this eval group
        num_subm_rows = 0
        subm_tbody = list(tbody_elem.iterfind('tr[2]/td/div/table/tbody'))[0]
        for row_indx, row in grouped.get_group(group).iterrows():
            if num_subm_rows < max_subm_rows:
                subm_tr = copy.deepcopy(template_subm_tr)
                for indx, elem in enumerate(subm_tr.iterfind('td')):
                    elem.text =  str(row[ subm_group_cols[indx] ])
                subm_tbody.append(subm_tr)
                num_subm_rows += 1
        return tbody_elem
        
    
    num_eval_rows = 0
    max_eval_rows = 20
    eval_group_cols = ['submission','eval_correctness','eval_frequency','eval_count']
    subm_group_cols = ['submission','subm_correctness','subm_frequency','subm_count']
    for group in groups:
        tbody = copy.deepcopy(template_tbody)
        if num_eval_rows <20:
            table.append(populate_tbody(group,tbody,eval_group_cols,subm_group_cols))
            num_eval_rows += 1
        else:
            break
    
    if output_html_path is None:
        output_html_path = "gui/" + "problem/" + os.path.basename(input_csv_path)[:-4] + ".html"
    #Check if output directory exists. If it doesn't, create it. 
    ensure_dir(output_html_path)
    
    with open(output_html_path,'w') as f:
        ET.ElementTree(root).write(f,pretty_print=True,method="html")
    
    return

def feedback_score(summary_df):
    '''
    Make a probability distribution for the wrong answers only, then calculate its L2 norm.
    '''
    # probability distribution for a particular evaluation group to be wrong.
    eval_cols = ['eval.0']+[val for val in summary_df.columns.values if val.startswith('eval.')]
    summary_df['eval.0'] = summary_df.index
    df = summary_df.drop_duplicates(subset=eval_cols)
    p_wrong = df['eval_frequency']*(1-df['eval_correctness']) / sum( df['eval_frequency']*(1-df['eval_correctness']) )
    rms_p_wrong = numpy.sqrt( sum(p_wrong*p_wrong) )
    return "{0:.2f}".format(rms_p_wrong)

def grading_score(summary_df):
    '''
    returns [num_correct,num_partial]
    where
    num_correct = number of evaluation groups whose submissions are 100% correct
    num_partial = number of evaluation group some but not all of whose submissions are correct
    '''
    #restore multi-index
    eval_cols = ['eval.0'] +  [val for val in summary_df.columns.values if val.startswith('eval.')]
    summary_df = summary_df.set_index(keys = eval_cols[1:], append=True)
    #group by evals, get group names, and sort by eval_count
    grouped = summary_df.groupby(level=eval_cols,sort=False)
    summary = grouped.aggregate({
                 'eval_correctness':numpy.mean,
                },sort=False)
    num_correct = sum(summary['eval_correctness']==1.0)
    num_partial = sum((summary['eval_correctness'] < 1.0)&(summary['eval_correctness'] > 0.0))
    
    return [str(num_correct),str(num_partial)]
    
    
def make_gui_index(problem_summary_dir = "problem_summary/",min_submissions=50):
    
    end_label = "_evaluated_summary.csv"
    problem_summary_list = []
    for filename in os.listdir(problem_summary_dir):
        if filename.endswith(end_label):
            problem_summary_list += [filename,]
    
    
    #########################
    #Import HTML Template
    #########################
    template_path = "gui/templates/gui_index_template.html"
    with open(template_path,'r') as f:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(f,parser)
    root = tree.getroot()
    #Get tbody from template. This is where data goes.
    tbody = list(tree.iterfind('body/table/tbody'))[0]
    template_tr = list(tbody.iterfind('tr'))[0]
    tbody.remove(template_tr)
    
    for problem_summary_filename in problem_summary_list:
        problem_name = problem_summary_filename[:-len(end_label)]
        problem_summary_csv_path = problem_summary_dir+problem_summary_filename
        problem_summary_html_path = 'problem/'+problem_name+'_evaluated_summary.html'
        summary_df = pandas.DataFrame.from_csv(problem_summary_csv_path,sep="\t")
        if sum(summary_df['subm_count']) < min_submissions:
            print("Not including " + problem_name + "in index, fewer than " + str(min_submissions) + " submissions" )
            continue
        #Make problem table
        make_gui_problem_table(problem_summary_csv_path)
        #Populate row for index
        tr = copy.deepcopy(template_tr)  
        td_list = list(tr.iterfind('td'))
        link = ET.SubElement(td_list[0],'a',attrib={'href':problem_summary_html_path })
        link.text = problem_name
        td_list[1].text = feedback_score(summary_df)
        [num_correct,num_partial] = grading_score(summary_df)
        td_list[2].text = num_correct
        td_list[3].text = num_partial
        td_list[4].text = str(sum(summary_df['subm_count']))
        td_list[5].text = summary_df['response_type'].iloc[0]
        #add row to index
        tbody.append(tr)
    
    output_html_path = "gui/gui.html"
    with open(output_html_path,'w') as f:
        ET.ElementTree(root).write(f,pretty_print=True,method="html")
        
##################################################
#       Loops (Not as carefully written)
##################################################  
import time
def run_eval_on_raw(raw_data_folder,case_sensitive=False,n_evals=5):     
    full_problem_list = []
    for filename in os.listdir(raw_data_folder):
        if filename.endswith(".csv"):
            full_problem_list += [filename,]
    
    counter = 0
    for filename in full_problem_list:
        start = time.time()
        print("\n"+"#"*100 + "\n" + "working on: "+ filename)
        try:
            make_eval_csv(raw_data_folder+"/"+filename,n_evals=n_evals,case_sensitive=case_sensitive)
            dur = round( (time.time()-start)/60.0 , 2)
            print("#"*10+" "*10+"Success! Problem:"+str(counter)+"/"+str(len(full_problem_list))+"!"+" Runtime: "+str(dur)+"min"+" "*10+"#"*10 )
        except ParseException:
            dur = round( (time.time()-start)/60.0 , 2)
            print("Could not parse submissions in file: "+filename + "Runtime: "+str(dur)+"min")
        except Exception as e:
            print("!"*50)
            print(e)
            print("!"*50)
        
        counter += 1
            

def run_summary_on_eval(eval_data_folder,n_sig=2):     
    full_problem_list = []
    for filename in os.listdir(eval_data_folder):
        if filename.endswith(".csv"):
            full_problem_list += [filename,]
    
    counter = 0
    for filename in full_problem_list:
        start = time.time()
        print("\n"+"#"*100 + "\n" + "working on: "+ filename)
        try:
            make_summary_csv(eval_data_folder+"/"+filename,n_sig=n_sig)
            dur = round( (time.time()-start)/60.0 , 2)
            print("#"*10+" "*10+"Success! Problem:"+str(counter)+"/"+str(len(full_problem_list))+"!"+" Runtime: "+str(dur)+"min"+" "*10+"#"*10 )
        except Exception as e:
            print("!"*50)
            print(e)
            print("!"*50)
        
        counter += 1


##################################################
#       Run Stuff
##################################################  
problem_checks_augmented_filename = "MITx_FakeCourse_problem_checks_augmented.csv"
problem_id_front = "i4x-MITx-FakeCourse-problem-"

raw_data_folder = "problem_check"
eval_data_Folder = "problem_evaluated"

split_csv_by_problem_id(problem_checks_augmented_filename,problem_id_front,acceptable_response_types=['formularesponse','numericalresponse'])
run_eval_on_raw(raw_data_folder,n_evals=5,case_sensitive=True)
run_summary_on_eval(eval_data_Folder)
make_gui_index(min_submissions=10)

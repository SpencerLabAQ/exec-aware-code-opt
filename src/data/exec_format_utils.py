import re

def label_line_cov(count):
    if count > 0:
        return "<e>"
    else:
        return ""

def quantize_exec_count(count):
    '''
    Quantization of line execution counts. The criteria we used considers the 
    distribution of the value counts.

    Stats: 
    Total num of lines: 3833623
    Min executions: 0 | Max executions: 1755
    Q1 (without 0 and 1): 3.0
    MEDIAN (without 0 and 1): 5.0
    Q3 (without 0 and 1): 8.0
    IQR (without 0 and 1): 5.0
    Outlier threshold (Q3 + 2,5 * IQR): 20.5

    Tokens:
    -       : Range = (-1.0, 0.0],      No of lines: 1832267 (48%)  [Line not executed]
    <e>     : Range = (0.0, 1.0],       No of lines: 1393089 (36%)  [Line executed exactly once]
    <e+>    : Range = (1.0, 5.0],       No of lines: 351667 (9%)
    <E>     : Range = (5.0, 20.5],      No of lines: 193992 (5%)
    <E+>    : Range = (20.5, 1755.0],   No of lines: 62608 (2%)
    '''

    assert count >= 0, "The count of line executions has to be a positive number"
    
    if count == 1: 
        return "<e>"
    elif count > 1 and count <= 5:
        return "<e+>"
    elif count > 5 and count <= 20.5:
        return "<E>"
    elif count > 20.5:
        return "<E+>"
    # zero case 
    return ""

def label_branch_exec(branch_exec_val):
    '''
    Three special tokens have been used to label branch coverage:
    - <NB>: Not a branch
    - <BC>: Branch covered in trace
    - <BNC>: Branch not covered in trace
    '''

    if branch_exec_val == None:
        return ""
    elif branch_exec_val == True:
        return "<BC>"
    elif branch_exec_val == False:
        return "<BNC>"
    else:
        raise ValueError(f"'{branch_exec_val}' value not allowed for branch coverage")

def remove_comments(text):
    # The pattern /\*.*?\*/ matches /* followed by any characters including newlines, ending with */
    pattern = r'/\*.+?\*/'
    # re.sub removes the pattern from the text, re.DOTALL allows . to match newlines
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text
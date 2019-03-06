
## The following is a module of function used to display the results the
## panel selection and control estimator contained in psc.ipynb.

import numpy as np
import ipywidgets as ipw
from IPython.display import display, display_html
import matplotlib.pyplot as plt
from matplotlib import gridspec
import itertools as itr
import pandas as pd
import json
import kernel as kr

#########################################
def psc_load(input_filename):
    """
INPUTS
input_filename        (string) short filename for estimates of coefficients

OUTPUT
out                   (list) of the following elements
  out[0]              (dict) estimator input dictionary
  out[1]              (df) estimates of each beta coeff
  out[2]              (df) summary of beta coeff estimates
  out[3]              (list of df's) estimates of each alpha coeff
  out[4]              (list of df's) summary of alpha coeff estimates
    """

    # Google Folder with estimation data
    output_folder = './est_out/'
    # Constructing the full data file name
    input_file_full = ''.join([output_folder,input_filename])
    # Loading the json
    with open(input_file_full) as f_obj:
        pscdata = json.load(f_obj)
    # Extracting the estimator input dictionary
    inpt_dic = pscdata[0].copy()
    del inpt_dic['inst_partition']
    del inpt_dic['panel_partition']
    del inpt_dic['inc']
    del inpt_dic['in_nm']
    del inpt_dic['dep_nm']
    del inpt_dic['en_nm']
    del inpt_dic['ex_nm']
    del inpt_dic['cin']
    del inpt_dic['tin']
    del inpt_dic['sec_pan']
    # Extracting Centered beta estimates
    bt_rs = pscdata[1]
    # df construction
    bt_df = pd.DataFrame(bt_rs['beta_cntrd'] ).T
    # Adding variable names
    bt_df.columns = bt_rs['beta_sum_clmn']
    # Extracting beta summary information
    bt_sm = pd.DataFrame(bt_rs['beta_sum_dat'])
    # Adding row bias,variance,mse names
    bt_sm.columns = bt_rs['beta_sum_clmn']
    # Adding variable names
    bt_sm.index = bt_rs['beta_sum_row']
    # Extracting Centered alpha estimates
    al_rs = pscdata[2]
    # df Construction
    al_ldf = [pd.DataFrame(al_rs['alpha_cntrd'][j]).T
              for j in range(len(al_rs['alpha_cntrd']))]
    # Extracting  alpha summary information
    al_sm = [pd.DataFrame(al_rs['alpha_sum_dat'][j])
             for j in range(len(al_rs['alpha_sum_dat']))]
    # Adding variable names and row names to df and summary df
    for j in range(len(al_rs['alpha_sum_dat'])):
        al_ldf[j].columns = al_rs['alpha_sum_clmn'][j]
        al_sm[j].columns = al_rs['alpha_sum_clmn'][j]
        al_sm[j].index= al_rs['alpha_sum_row']
    # Collecting into output list
    out = [inpt_dic, bt_df, bt_sm, al_ldf, al_sm]
    return out

#########################################

def pscsum_load(input_filename):
    """
INPUTS
input_filename        (string) short filename for summary of dgp

OUTPUTS
out                   (list of lists) of the following elements
  out[0]                (dict) Input dictionary for data set generation
  out[1]                (df) True beta coefficients
  out[2]                (df) True alpha coefficients
    """

    # Google folder with dgp data
    data_folder = './data_sum/'
    # Full filename for the dgp summary
    data_file_full = ''.join([data_folder,input_filename])
    # Loading the json
    with open(data_file_full) as f_obj:
        pscdata = json.load(f_obj)
    # Extracting the data set dgp dictionary
    inpt_dict = pscdata[0].copy()
    del inpt_dict['dep_nm']
    del inpt_dict['en_nm']
    del inpt_dict['ex_nm']
    del inpt_dict['cin']
    del inpt_dict['tin']
    del inpt_dict['sec_pan']
    del inpt_dict['frc']
    # Names of primary coefficients
    pcoeff_nms = ([ ''.join(['$\\beta_{',str(1),',',str(i+1),'}$'])
                        for i in range(pscdata[0]['n_end'])]
                 +[ ''.join(['$\\beta_{',str(2),',',str(i+1),'}$'])
                        for i in range(pscdata[0]['n_exo'])])
    # Extracting the true primary coefficients
    pcoeff = pd.DataFrame(pscdata[1]['pcoeff']).T
    # Adding names of each primary coefficient
    pcoeff.columns = pcoeff_nms
    # Names of alpha coefficients
    coeff_nms = [([ ''.join(['$\\alpha_{',str(j+1),str(1),',',str(i+1),'}$'])
                       for i in range(pscdata[0]['n_exo'])]
                  + [''.join(['$\\alpha_{',str(j+1),str(2),',',str(i+1),'}$'])
                       for i in range(pscdata[0]['t_inst'])])
                 for j in range(pscdata[0]['n_end'])]
    # Adding name of each alpha coefficient
    coeff = [pd.DataFrame(pscdata[1]['coeff'][i],columns = coeff_nms[i])
                        for i in range(len(pscdata[1]['coeff']))]
    # Constructing the out put dictionary
    out = [inpt_dict , pcoeff , coeff ]

    return out

#########################################

def coeffden(coeff,line_nms,x_lm,y_lm,c_h,w,n_eqn,s_eqn):
    """
INPUTS
ceoff            (list of df's)  input data
line_nms         (list of strings) legend entry names
x_lm             (list of int) lower and upper x limits of the plot
y_lm             (int) upper y limit of the plot
c_h              (int) Constant in plug in bandwidth calculation
w                (int) plotted variable column number
n_eqn            (int) number of equations per cross section
s_eqn            (int) indicator for which equation to plot

OUTPUTS
out              (plot) density plot
    """
    # Number of data set different run data sets in coeff
    nds = len(coeff)
    # Resetting coefficient number to index number
    w = w-1
    # Resetting equation number to index number
    s_eqn = s_eqn-1
    # Extracting the coefficient for the correct equation number
    if n_eqn > 1:
        coeff = [coeff[i][s_eqn] for i in range(nds)]
    # Extracting the number of coeffients
    ncfs = coeff[0].shape[0]
    # Closing all open plots in current cell
    plt.close('all')
    # Trimmed coefficient lists
    coeff_trimd=[]
    for i in range(nds):
        coeff_vals = coeff[i].iloc[:,w].values
        coeff_logical = np.logical_and(coeff_vals>np.percentile(coeff_vals,2)
                                  ,coeff_vals<np.percentile(coeff_vals,98)).tolist()
        coeff_trimd.append(coeff[i].iloc[coeff_logical,w])
    # Converting to np arrary and sorting the values in coeff
    a = [ np.sort(coeff_trimd[i].values,axis = 0) for i in range(nds) ]
    # Caluculating plug in bandwidths
    h = [ c_h*ncfs**(-1/5)*np.std(a[i]) for i in range(nds)]
    # Calculating the density of sorted coefficients
    aden = [ kr.mvden(a[i],a[i],h[i],9).reshape(len(a[i]),1) for i in range(nds) ]
    # The rest plotting with matplotlib
    f,ax = plt.subplots()
    f.set_figheight(7)
    f.set_figwidth(15)
    ax.set_xlim((x_lm[0],x_lm[1]))
    ax.set_ylim((0,y_lm))
    for i in range(nds):
        ax.plot(a[i],aden[i])
    ax.legend(line_nms)
    ax.grid(which = 'both')
    ax.set_title(''.join(['Distribution of Estimated ',coeff[0].columns[w]]))
    plt.show()

########################################
def display_side_by_side(dfs,nms):
    html_str = ''
    html_str+= '<table>'
    for j in range(len(dfs)):
        html_str+= '<td>'
        html_str+= nms[j]
        html_str+= '<br>'
        html_str+= dfs[j].to_html()
        html_str+= '<td>'
    html_str+='<table>'
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

#########################################

def tbl_dsp(tables,n_eqn,s_eqn,line_nms,horz = 0):
    """
INPUTS
tables           (list of df's)  summary table's
line_nms         (list of strings) case number names
n_eqn            (int) number of equations per cross section
s_eqn            (int) indicator for which equation's table to output

OUTPUTS
A number of summary tables displayed

    """
    # Shifting the value of the indicator to match index
    s_eqn = s_eqn-1
    # Outputing Tables
    if horz == 0:
        if n_eqn > 1:
            for j in range(len(tables)):
                display(tables[j][s_eqn])
                display(''.join(['Case ', str(j+1),':', line_nms[j]]))
        elif n_eqn == 1:
            for j in range(len(tables)):
                display(tables[j])
                display(''.join(['Case ', str(j+1),':', line_nms[j]]))
    elif horz == 1:
        display_side_by_side(tables,line_nms)

#########################################

def cfs_dsp(coeff,tables,n_eqn,y_lm,line_nms,tbl_horz = 0):
    """
INPUTS
tables           (list of df's)  summary table's
ceoff            (list of df's)  input data
line_nms         (list of strings) legend entry names
y_lm             (int) upper y limit of the plot

OUTPUTS
Plot and Tables Displayed with interactive widgets

    """
    # Layout of each widget
    box_hlayout = ipw.Layout(display='flex',flex_flow='row',align_items='stretch'
                             ,width='95%')

    box_vlayout = ipw.Layout(display='flex', flex_flow='column', align_items='stretch',
                    width='10%', height = 'auto', justify_content='space-between')

    if n_eqn == 1:
        # Coefficient selection widget
        coeff_sel = ipw.IntSlider( min = 1 , max = coeff[0].shape[1], value = 1, step = 1
                             , description = 'Coefficient:'
                              ,width = 'auto',layout = box_hlayout
                              ,style = {'description_width': 'initial'})
    elif n_eqn > 1:
        # Coefficient selection widget
        coeff_sel = ipw.IntSlider( min = 1 , max = coeff[0][0].shape[1], value = 1, step = 1
                              ,description = 'Coefficient:'
                              ,width = 'auto',layout = box_hlayout
                              ,style = {'description_width': 'initial'})
    # Equation selection widget
    eqn_sel = ipw.IntSlider( min = 1 , max = n_eqn, value = 1, step = 1
                              ,description = 'Equation:'
                              ,width = 'auto',layout = box_hlayout
                              ,style = {'description_width': 'initial'})
    # PU bandwidth constant selection widget
    ch_sel = ipw.FloatSlider(min = 0.1 , max = 3, value = 2.5, step =0.2
                              ,description = 'Bandwidth Constant'
                              ,width = 'auto',layout = box_hlayout
                              ,style = {'description_width': 'initial'})
    # Horizontal display range slider widget
    xlim_sel = ipw.FloatRangeSlider( value=[-0.4, 0.4], min=-1,max=1, step=0.05
                                    ,description='x limits:'
                                    ,disabled=False ,continuous_update=False
                                    ,orientation='horizontal',readout=True
                                    ,readout_format='.1f',width = 'auto'
                                    ,layout=box_hlayout
                                    ,style = {'description_width': 'initial'})

    ylim_sel = ipw.FloatSlider(min = 0 , max = 15, value = y_lm, step = 1
                     ,description = 'y limits'
                     ,orientation = 'vertical',length = 'auto'
                     ,layout = box_vlayout
                     ,style = {'description_length': 'initial'},readout = False)

    # Interactive call of density function plot
    coeff_out =  ipw.interactive_output(coeffden ,{'coeff': ipw.fixed(coeff)
                                                 ,'line_nms': ipw.fixed(line_nms)
                                                 ,'x_lm': xlim_sel, 'y_lm': ylim_sel
                                                 ,'c_h': ch_sel,'w': coeff_sel
                                                 ,'n_eqn': ipw.fixed(n_eqn)
                                                 ,'s_eqn': eqn_sel})
    # Interactive call of table display function
    table_out = ipw.interactive_output(tbl_dsp,{'tables': ipw.fixed(tables)
                                                ,'n_eqn': ipw.fixed(n_eqn)
                                                ,'s_eqn': eqn_sel
                                                ,'line_nms': ipw.fixed(line_nms)
                                                ,'horz': ipw.fixed(tbl_horz)})
    # Return of the constructed block with widgetes and tables.
    if n_eqn == 1:
        return ipw.VBox([table_out,ipw.HBox([coeff_out,ylim_sel]),coeff_sel,ch_sel,xlim_sel])
    else:
        return ipw.VBox([table_out,ipw.HBox([coeff_out,ylim_sel]),coeff_sel,ch_sel,xlim_sel,eqn_sel])

#########################################

def indict_dsp(inpt,depth):
    """
INPUTS
inpt            (list of (lists) of dicts) diction
depth           (int) number of df

OUTPUTS
displayed df's with widgets

    """
    w_sel = ipw.IntSlider( min = 1 , max = len(inpt), value = 1, step = 1
                                 , description = 'Results Dataset: '
                                  ,width = 'auto', style = {'description_width': 'initial'})
    d_sel = ipw.IntSlider( min = 1 , max = depth , value = 1, step = 1
                                 , description = 'Equation: '
                                  ,width = 'auto', style = {'description_width': 'initial'})
    def in_dcts_dsp(in_dcts,depth,w,d):
        if depth == 1:
            display(in_dcts[w-1])
        elif depth > 1:
            display(in_dcts[w-1][d-1])

    dict_out = ipw.interactive_output(in_dcts_dsp,{'in_dcts': ipw.fixed(inpt),'w': w_sel })

    if depth == 1:
        dict_out = ipw.interactive_output(in_dcts_dsp,{'in_dcts': ipw.fixed(inpt)
                                                       ,'w': w_sel
                                                       ,'depth': ipw.fixed(depth)
                                                       ,'d': ipw.fixed(1)})
        bx = ipw.VBox([w_sel,dict_out])
    elif depth > 1:
        dict_out = ipw.interactive_output(in_dcts_dsp,{'in_dcts': ipw.fixed(inpt),'w': w_sel
                                                       ,'depth': ipw.fixed(depth),'d': d_sel})
        bx =ipw.VBox([d_sel,w_sel,dict_out])

    return bx

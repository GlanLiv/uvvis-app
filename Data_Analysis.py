import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from PIL import Image
from datetime import date, datetime, timedelta
from streamlit.web.server.websocket_headers import _get_websocket_headers
from bokeh.plotting import figure, output_file, show
import peakutils
from bokeh.models import Label
from io import StringIO 

from st_pages import Page, show_pages, add_page_title, hide_pages

# Optional -- adds the title and icon to the current page
#add_page_title()

img_icon = Image.open("./icon.jpg")
st.set_page_config(
   page_title="gadget HLS",
   page_icon=img_icon,
)

# Specify what pages should be shown in the sidebar
#show_pages(
#    [
#      Page("gadget_app.py", "HLS Analytical Equipment"),
#      Page("pages/Data Analysis.py", "Data Analysis"),
#    ]
#)



#####  FHNW LOGO ##########

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-color: yellow;
                background-image: url("https://webteam.pages.fhnw.ch/fhnw-styleguide-v4/assets/img/fachhochschule-nordwestschweiz-fhnw-logo.svg");
                background-size: 295px;
                #width: 400px;
                #height: 300px;
                background-repeat: no-repeat;
                padding-top: 40px;
                background-position: 20px 40px;
            }
            #[data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            #}
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

############# Analytical Method Selection #########

#st.header('VORSICHT!: Diese Seite wird gerade gewartet! PRECAUTION!: This page is under construction!')

option = st.sidebar.selectbox(
    'Select an analytical method:',
    ('MALDI', 'UV-vis', 'LC-MS'))

if option =='MALDI' or option =='LC-MS':
 option_MALDI = st.sidebar.selectbox('Select a data process method:',
    ('Spectrum Normalization', 'Peptide Assignment after Tryptic Digestion of Protein'))

 raw_data = st.sidebar.file_uploader('Select your text files to upload', accept_multiple_files=True, type={"csv", "txt"})


 container = st.container()
 container2 = st.container()


###########################  MALDI #########################

 thres = 'start'
 min_dist = 'start'
 xmin = 'start'
 xmax = 'start'

 def get_x_range(a,b,c,d):
  number_of_spectra = len(raw_data)
  spectra_count = 0
  #spec=0

  while raw_data is not None:

   if spectra_count < number_of_spectra:
    if a != 'start' or b != 'start' or c != 'start' or d != 'start':
     raw_data[spectra_count].seek(0)
    else:
     pass
    if option=='LC-MS':
     spec = pd.read_csv(raw_data[spectra_count], sep='[;, \t\s+]', engine='python', header=None, encoding='utf-16', encoding_errors='ignore') 
    elif option=='MALDI':
     spec = pd.read_csv(raw_data[spectra_count], sep='[;, \t\s+]', engine='python', header=None) 
    else:
     spec= pd.read_csv(raw_data[spectra_count], sep='[;, \t\s+]', engine='python', header=None) 

   else:
    return spec
    break


   spectra_count += 1
   continue 

 def reading_in_raw_data(a,b,c,d):
  p=[]
  p_y=[]
  spectrum = figure(
    #title='simple line example',
    x_axis_label='m/z',
    y_axis_label='counts (arb. units)')
 
  number_of_spectra = len(raw_data)
  spectra_count = 0

  while raw_data is not None:
   if spectra_count == number_of_spectra: 
    return spectrum, p, p_y
    break
 
   if spectra_count < number_of_spectra:
     #if a != 'start' or b != 'start' or c != 'start' or d != 'start':
     raw_data[spectra_count].seek(0)
     #else:
      #pass
     if option=='LC-MS':
      spectra_df = pd.read_csv(raw_data[spectra_count], sep='[;, \t\s+]', engine='python', header=None, encoding='utf-16', encoding_errors='ignore') 
     elif option=='MALDI':
      spectra_df = pd.read_csv(raw_data[spectra_count], sep='[;, \t\s+]', engine='python', header=None) 
     else:
      spectra_df = pd.read_csv(raw_data[spectra_count], sep='[;, \t\s+]', engine='python', header=None) 
     data_x = spectra_df[0]
     data_y = spectra_df[1]


     if c != 'start' or d != 'start':
      index_xmin = min(range(len(data_x)), key=lambda i: abs(data_x[i]-[c]) )
      index_xmax = min(range(len(data_x)), key=lambda i: abs(data_x[i]-[d]) )

      y_max=max(data_y[index_xmin:index_xmax])
      data_y = data_y/y_max
     else:
      pass

     if a != 'start' or b != 'start':
      print('inputpeaks')
      indeces = peakutils.indexes(data_y, thres=float(a), min_dist=int(b))
      for i in range(len(indeces)):
       p = np.append(p, round(data_x[indeces[i]]))
       p_y = np.append(p_y, round(data_y[indeces[i]]))
       annotation = Label(x=data_x[indeces[i]], y=data_y[indeces[i]], text=str(data_x[indeces[i]]))
       if len(indeces) > 1:
        spectrum.add_layout(annotation)
      if len(indeces) > 1:
       container2.write(p)
     else:
      print('noinput')
      pass

   color=['black','darkred','darkblue','darkgreen', 'red', 'blue', 'green', 'orange'] 
   spectrum.line(data_x, data_y, legend_label=raw_data[spectra_count].name[0:-4], line_width=1.5, color=color[spectra_count])
  
   spectra_count += 1
   continue



if option == 'MALDI' and len(raw_data) !=0 or option=='LC-MS' and len(raw_data) !=0: 

 if option_MALDI=='Spectrum Normalization': 
  value_y=1.0
  value_x=10000
  spectra_df = get_x_range(thres, min_dist, xmin, xmax)
  data_x = spectra_df[0][:]
  data_y = spectra_df[1][:]

 if option_MALDI=='Peptide Assignment after Tryptic Digestion of Protein':
  value_y=0.10
  value_x=150
  spectra_df = get_x_range(thres, min_dist, xmin, xmax)
  data_x = spectra_df[0][:]
  data_y = spectra_df[1][:]

##################### manipulation tools ################

 st.sidebar.subheader('Select normalization region')
 xmin = st.sidebar.number_input('m/z min', min_value=float(min(data_x)), max_value=float(max(data_x)), value=float(min(data_x)))
 xmax = st.sidebar.number_input('m/z max', min_value=float(min(data_x)), max_value=float(max(data_x)), value=float(max(data_x)))
 
 st.sidebar.subheader('Select parameters for peak annotation')
 thres = st.sidebar.number_input('threshold in y', min_value=0.0, max_value=1.0, step=0.1, value=value_y)
 min_dist = st.sidebar.number_input('threshold in x', min_value=1, max_value=10000, value=value_x)

 
 spectrum, peaks, peaks_y = reading_in_raw_data(thres, min_dist, xmin, xmax)

 if option_MALDI=='Peptide Assignment after Tryptic Digestion of Protein':
  seq = st.sidebar.text_input('Enter Amino Acid Sequence of Protein (N -> C):')
  #if seq[3] == '-':
   #for s in seq:
     

  ## residue mass, amu (monoisotopic) (alle Massen sind mit -H2O)
  amino_dic = {'G': 57.02,
                'A': 71.04,
                'S':87.03,
                'P':97.05,
                'V':99.07,
                'T':101.05,
                'C':103.01,
                'I':113.08,
                'L':113.08,
                'N':114.04,
                'D':115.03,
                'Q':128.06,
                'K':128.09,
                'E':129.04,
                'M': 131.04,
                'H':137.06,
                'F':147.07,
                'R':156.10,
                'Y':163.06,
                'W':186.08,
                }
  
  def check_ions(a,d):
      diff =4 
      mem = 0
      print(peaks)
      for p in range(len(peaks)):
        if diff > a-peaks[p] > -diff:
          st.write(':green[MH+-ion of sequence %d found in peaklist]' % (d))
          st.write(':green[MH+-ion mass of sequence %d = %d, peak = %d]' % (d, a, peaks[p]))
          MH_x=peaks[p]
          MH_y=peaks_y[p]
          annotation = Label(x=MH_x, y=MH_y-(MH_y)*0.1, text='sequence %d' % (d))
          #spectrum.add_layout(annotation)
          mem=1
      if mem==0:
          if 3500 > seqH > 600:
           st.write(':red[NOT found in peaklist]')
          else:
           st.write(':orange[Mass might be out of range!]')

  def check_ions_Na(a,d):
      diff =4 
      mem = 0
      print(peaks)
      for p in range(len(peaks)):
        if diff > a-peaks[p] > -diff:
          st.write(':green[MNa+-ion of sequence %d found in peaklist]' % (d))
          st.write(':green[MNa+-ion mass of sequence %d = %d, peak = %d]' % (d, a, peaks[p]))
          MH_x=peaks[p]
          MH_y=peaks_y[p]
          annotation = Label(x=MH_x, y=MH_y-(MH_y)*0.1, text='sequence %d' % (d))
          #spectrum.add_layout(annotation)
          mem=1
      if mem==0:
          if 3500 > seqH > 600:
           st.write(':red[NOT found in peaklist]')
          else:
           st.write(':orange[Mass might be out of range!]')


  mass_seq = 0
  seq_name = ' '
  seq_count = 0
  print(len(seq))
  for amino in range(len(seq)):
     if seq[amino] == 'K' or seq[amino] == 'R':
       print(mass_seq)
       print(amino_dic[seq[amino]])
       mass_seq += amino_dic[seq[amino]]
       seq_name += seq[amino]
       mass_seq += 18
       seq_count +=1
       seqH = mass_seq+1
       seqNa = mass_seq+23
       seqK=mass_seq+39
       st.write('sequence %d (%d-%d) is %s (MH+=%d)' % (seq_count, amino+3-len(seq_name), amino+1, seq_name, seqH))
       check_ions(seqH, seq_count)
       st.write('sequence %d (%d-%d) is %s (MNa+=%d)' % (seq_count, amino+3-len(seq_name), amino+1, seq_name, seqNa))
       check_ions_Na(seqNa, seq_count)

       if 'S' or 'T' in seq_name:
        seqH -= 18
        seqNa -= 18
        seqK -= 18
        st.write('checking for hydrolyzed forms of S and T...')
        check_ions(seqH, seq_count)
        st.write('checking for hydrolyzed forms of S and T...')
        check_ions_Na(seqNa, seq_count)

       if 'M' in seq_name:
        seqH -= 48
        seqNa -= 48
        seqK -= 48
        st.write('checking for hydrolized forms of M...')
        check_ions(seqH, seq_count)
        st.write('checking for hydrolized forms of M...')
        check_ions_Na(seqNa, seq_count)
 

       mass_seq = 0
       seq_name = ' '   
     else:
       print('else')
       print(mass_seq)
       print(seq[amino])
       print('done')
       mass_seq += amino_dic[seq[amino]]
       seq_name += seq[amino]


 
 def clear():
   thres = 1
   min_dist = 1000
   xmin = min(spectra_df[0])
   xmax = max(spectra_df[0])
   #reading_in_raw_data(thres, min_dist, xmin, xmax)

 #st.button('reset', on_click=clear())
 
 spectrum.legend.click_policy = "hide"
 output_file("interactive_legend.html")
 spectrum.yaxis.visible= False
 container.bokeh_chart(spectrum, use_container_width=False)

#MHHHHHHHHHHLEVLFQGPSPDTTSLNIADDVRMDPRLKAMLAAFPMMEQQTFQTREEQVANANTPEATAAREQLKMMMDMMDSEEFAPSDNLDISTREFTSSPDGNAIKIQFIRPKGKQKVPCVYYIHGGGMMIMSAFYGNYRAWGKMIANNGVAVAMVDFRNCLSPSSAPEVAPFPAGLNDCVSGLKWVSENADELSIDKNKIIIAGESGGGNLTLATGLKLKQDGNIDLVKGLYALCPYIAGKWPQDRFPSSSENNGIMIELHNNQGALAYGIEQLEAENPLAWPSFASAEDMQGLPPTVINVNECDPLRDEGIDFYRRLMAAGVPARCRQVMGTCHAGDMFVAVIPDVSADTAADIARTAKGGA
#Lactoferrin
#MKLVFLVLLFLGALGLCLAGRRRSVQWCAVSQPEATKCFQWQRNMRKVRGPPVSCIKRDSPIQCIQAIAENRADAVTLDGGFIYEAGLAPYKLRPVAAEVYGTERQPRTHYYAVAVVKKGGSFQLNELQGLKSCHTGLRRTAGWNVPIGTLRPFLNWTGPPEPIEAAVARFFSASCVPGADKGQFPNLCRLCAGTGENKCAFSSQEPYFSYSGAFKCLRDGAGDVAFIRESTVFEDLSDEAERDEYELLCPDNTRKPVDKFKDCHLARVPSHAVVARSVNGKEDAIWNLLRQAQEKFGKDKSPKFQLFGSPSGQKDLLFKDSAIGFSRVPPRIDSGLYLGSGYFTAIQNLRKSEEEVAARRARVVWCAVGEQELRKCNQWSGLSEGSVTCSSASTTEDCIALVLKGEADAMSLDGGYVYTAGKCGLVPVLAENYKSQQSSDPDPNCVDRPVEGYLAVAVVRRSDTSLTWNSVKGKKSCHTAVDRTAGWNIPMGLLFNQTGSCKFDEYFSQSCAPGSDPRSNLCALCIGDEQGENKCVPNSNERYYGYTGAFRCLAENAGDVAFVKDVTVLQNTDGNNNEAWAKDLKLADFALLCLDGKRKPVTEARSCHLAMAPNHAVVSRMDKVERLKQVLLHQQAKFGRNGSDCPDKFCLFQSETKNLLFNDNTECLARLHGKTTYEKYLGPQYVAGITNLKKCSTSPLLEACEFLRK

################################# UV-vis ##########################

if option =='UV-vis':

 container3 = st.container()
 raw_data_analyte = st.sidebar.file_uploader('Select your analyte data (one file):', type={"csv", "txt"})
 raw_data = st.sidebar.file_uploader('Select your calibrant data (multiple files):', accept_multiple_files=True, type={"csv", "txt"})

 st.sidebar.write('This tool is created for txt/csv/ASCII files generated by the Agilent Cary 60 UV-vis machine. Still having troubles? Remove the headers and/or footers.')

 def reading_in_raw_data():
  spectrum = figure(
    title='UV-vis spectrum',
    x_axis_label='Wavelength (nm)',
    y_axis_label='Absorbance (arb. units)')
 
  number_of_spectra = len(raw_data)
  spectra_count = 0
  data_y_sum = np.zeros([180])
  data_x = np.zeros([180])

  if raw_data_analyte is not None:
     stringio = StringIO(raw_data_analyte.getvalue().decode("utf-8"))
     string_data = stringio.readlines()
     s_count=0
     h_count=0
     for string in range(len(string_data)):
      s_count += 1
      if 'Collection Time' in string_data[string]:
       print(s_count)
       s_count=len(string_data)-s_count+3
       break
      else:
       if string==len(string_data)-1:
        s_count=0
      if 'Wavelength (nm),Abs,' in string_data[string]:
       h_count=2

     print(s_count)
     spectra_df = pd.read_csv(raw_data_analyte, sep='[;, \t]', engine='python', skipinitialspace=True, skiprows=h_count, skipfooter=s_count, header=None, usecols=range(0,2))
     print('here')
     print(spectra_df)
     data_x_analyte = spectra_df[0]
     data_y = spectra_df[1]
     correction = np.full((len(data_y)), min(data_y))
     spectrum.line(data_x_analyte, np.array(data_y)-correction, legend_label=raw_data_analyte.name[0:-4], line_width=1.5, color='black')
     data_x=data_x_analyte

  while raw_data is not None:
   while True: 
    if spectra_count == number_of_spectra: 
     spectrum.line(data_x, data_y_sum, legend_label='sum spectrum', line_width=1.5, color='red')
     return spectrum
     break
 
    if spectra_count < number_of_spectra:
     ################# slider input #############
     factor = st.select_slider(value=1, options=np.arange(0.0,3.0, 0.01), key=str(spectra_count), label='factor for %s' % raw_data[spectra_count].name[0:-4]) 

 
     raw_data[spectra_count].seek(0)
     stringio = StringIO(raw_data[spectra_count].getvalue().decode("utf-8"))
     string_data = stringio.readlines()
     s_count=0
     h_count=0
     for string in range(len(string_data)):
      s_count += 1
      if 'Collection Time' in string_data[string]:
       s_count=len(string_data)-s_count+3
       break
      else:
       if string==len(string_data)-1:
        s_count=0
      if 'Wavelength (nm),Abs,' in string_data[string]:
       h_count=2

     spectra_df = pd.read_csv(raw_data[spectra_count], sep='[;, \t]', engine='python', skipinitialspace=True, skiprows=h_count, skipfooter=s_count, header=None, usecols=range(0,2))
     data_x_un = spectra_df[0]
     data_y_un = spectra_df[1]
     print(data_x_un)

     data_x = np.arange(start=220.0, stop=400.0, step=1.0)
     data_y = []
     for data in range(len(data_x)):
       index_x = min(range(len(data_x_un)), key=lambda i: abs(float(data_x_un[i])-data_x[data]) )
       data_y += [data_y_un[index_x]]

    correction = np.full(len(data_y), min(data_y))
    color=['darkred','darkblue','darkgreen', 'red', 'blue', 'green', 'orange'] 
    data_y = factor*(np.array(data_y)-correction)
    spectrum.line(data_x, data_y, legend_label=raw_data[spectra_count].name[0:-4], line_dash='dashed', line_width=1.5, color=color[spectra_count])
   
    data_y_sum += data_y

    spectra_count += 1
    continue

 spectrum = reading_in_raw_data()

 spectrum.legend.click_policy = "hide"
 output_file("interactive_legend.html")
 container3.bokeh_chart(spectrum, use_container_width=False)




############### footer ######################

ft = """
<style>
a:link , a:visited{
color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: #0283C3; /* theme's primary color*/
background-color: transparent;
text-decoration: underline;
}

#page-container {
  position: relative;
  min-height: 10vh;
}

header{
   visibility:hidden;
}

#MainMenu{
   visibility:hidden;
}

footer{
    visibility:hidden;
}

.footer {
position: relative;
left: 0;
top:230px;
bottom: 0;
width: full;
background-color: black;
color: #808080; /* theme's text color hex code at 50 percent brightness*/
text-align: center; /* you can replace 'left' with 'center' or 'right' if you want*/
}
</style>

<div id="page-container">

<div class="footer">
                <a href="https://www.fhnw.ch" target="_blank">www.fhnw.ch</a>
                <span class="d-inline-block mx-1 mx-md-2">|</span>
                <a href="https://gadget.lifesciences.fhnw.ch/Impressum" target="_blank">Impressum</a>
                <span class="d-inline-block mx-1 mx-md-2">|</span>
                <a href="https://gadget.lifesciences.fhnw.ch/Datenschutz" target="_blank">Datenschutz</a>
<p style='font-size: 0.875em;'>Made with <a style='display: inline; text-align: left;' href="https://streamlit.io/" target="_blank">Streamlit</a><br 'style= top:3px;'></p>
</div>

</div>
"""
st.write(ft, unsafe_allow_html=True)


import argparse
import textwrap
import pickle

# collect all the arguments from command line
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=textwrap.dedent('''\
    When choosing brain structure number, please input the number corresponding to the structure
        0 -> Total Brain Volume
        1 -> Average volume of Thalamus
        2 -> Average volume of Caudate
        3 -> Average volume of Putamen
        4 -> Average volume of Hippocampus
        5 -> Average volume of Amygdala
        6 -> Average volume of Accumbens
        7 -> Average volume of Pallidum
    
    When choosing adjustment method number, please input the number corresponding to the adjustment method
        0 -> proportion adjustment
        1 -> power-proportion adjustment
        2 -> general linear model adjustment
    
    Also note that when inputting sex, 0 corresponds to female and 1 corresponds to male.
    Also note that age is in years.
'''))
parser.add_argument("brain_structure_number", type=int, help = "The number associated with volume of the brain structure.", default = 0)
parser.add_argument("brain_structure_volume", type=float, help = "The volume of the brain structure.", default = 0)
parser.add_argument("adjustment_method_number", type=int, help = "The number associated with the adjustment method.", default = 0)
parser.add_argument("total_intracranial_volume", type=float, help = "The total intracranial volume.", default = 0)
parser.add_argument("-b", "--beta_value", type=float, help = "The beta value (used in power-proportion).", required = False, default = -1)
parser.add_argument("-s", "--sex", type=float, help = "The sex of the patient (used in glm adjustment).", required = False, default = 0)
parser.add_argument("-a", "--age", type=float, help = "The age of the patient (used in glm adjustment).", required = False, default = 0)
parser.add_argument("-bmi", type=float, help = "The BMI of the patient (used in glm adjustment).", required = False, default = 0)
parser.add_argument("-c", "--custom_file_name", type=str, help = "Specify a custom file name.", required = False, default = "")
args = parser.parse_args()

# parse arguments and assign to appropiate variables
volume_number = args.brain_structure_number
adjustment_number = args.adjustment_method_number
voi_volume = args.brain_structure_volume
tiv_volume = args.total_intracranial_volume
beta_value = args.beta_value
sex = args.sex
age = args.age
bmi = args.bmi
custom_file_name = args.custom_file_name



def adjustment_function(volume_number, adjustment_number, voi_volume, tiv_volume, beta_value, sex, age, bmi):
    
    file_output = ""

    brain_structures = {0:'Total Brain Volume', 1:'Average volume of Thalamus', 2:'Average volume of Caudate', 3:'Average volume of Putamen', 4:'Average volume of Hippocampus', 5:'Average volume of Amygdala', 6:'Average volume of Accumbens', 7:'Average volume of Pallidum'}

    print('You have chosen ' + brain_structures[int(volume_number)])

    adjustment_methods = {0:'proportion adjustment', 1:'power-proportion adjustment', 2:'general linear model adjustment'}

    print('You have chosen ' + adjustment_methods[int(adjustment_number)])

    if adjustment_number == 0:
        """
        proportion adjustment
        needs Volume of Interest and Total Intracranial Volume
        """
        
        adjusted_volume = float(voi_volume) / float(tiv_volume)
        print('The adjusted Volume is: ' + str(adjusted_volume) + 'mL')

        # append onto string
        file_output += 'Volume of Interest: ' + brain_structures[int(volume_number)] + '\n'
        file_output += 'Volume of Interest Volume: ' + str(voi_volume) + '\n'
        file_output += 'Total Intracranial Volume: ' + str(tiv_volume) + '\n'
        file_output += 'The adjusted Volume is: ' + str(adjusted_volume) + 'mL'

    if adjustment_number == 1:
        """
        power-proportion adjustment
        needs Volume of Interest and Total Intracranial Volume and fitted beta value
            - fitted beta value can be custom or the pre-fitted value
        """

        if beta_value == -1:
            custom_beta = 'n'
            beta_values = {0: 0.893836353, 1: 0.653000982, 2: 0.703072538, 3: 0.711833489, 4: 0.466289446, 5: 0.696672306, 6: 0.71295332, 7: 0.639609375}
            beta_value = beta_values[int(volume_number)]
        else:
            custom_beta = 'y'

        adjusted_volume = float(voi_volume) / (float(tiv_volume) ** beta_value)
        print('The adjusted Volume is: ' + str(adjusted_volume) + 'mL')

        # append onto string
        file_output += 'Volume of Interest: ' + brain_structures[int(volume_number)] + '\n'
        file_output += 'Volume of Interest Volume: ' + str(voi_volume) + '\n'
        file_output += 'Total Intracranial Volume: ' + str(tiv_volume) + '\n'
        file_output += 'Beta value: ' + str(beta_value) + '\n'
        file_output += 'Custom beta value?: ' + custom_beta + '\n'
        file_output += 'The adjusted Volume is: ' + str(adjusted_volume) + 'mL'

    if adjustment_number == 2:
        """
        general linear model adjustment
        needs Volume of Interest, Total Intracranial Volume, Sex, Age, BMI 
        """

        brain_structures_ = {'0':'Total_Brain_Volume', '1':'Thalamus', '2':'Caudate', '3':'Putamen', '4':'Hippocampus', '5':'Amygdala', '6':'Accumbens', '7':'Pallidum'}
        # Load model from file
        with open("pickle_model_" + brain_structures_[str(volume_number)] + ".pkl", 'rb') as file:
            regressor = pickle.load(file)

        intercept = regressor.intercept_

        # predict using given paramters
        # tiv needs to be normalised since the model uses normalised tiv
        average_tiv = 1548809.7428022022
        tiv_vol = float(tiv_volume) / average_tiv
        y_pred = regressor.predict([[float(sex), float(bmi), float(age), tiv_vol]])

        # calculate residual + intercept, represents new data
        residual = float(voi_volume) - y_pred
        adjusted_volume = residual + intercept
        print('The adjusted Volume is: ' + str(adjusted_volume[0][0]) + 'mL')

        # append onto string
        file_output += 'Volume of Interest: ' + brain_structures[int(volume_number)] + '\n'
        file_output += 'Volume of Interest Volume: ' + str(voi_volume) + '\n'
        file_output += 'Total Intracranial Volume: ' + str(tiv_volume) + '\n'
        if sex == "0":
            file_output += 'Sex: female' + '\n'
        else:
            file_output += 'Sex: male' + '\n'
        file_output += 'Age: ' + str(age) + '\n'
        file_output += 'BMI: ' + str(bmi) + '\n'
        file_output += 'The adjusted Volume is: ' + str(adjusted_volume[0][0]) + 'mL'

    # if a custom file name is not specified, use a default file name
    if custom_file_name == "":
        file_name = brain_structures[int(volume_number)].replace(' ', '_') + "_" + adjustment_methods[int(adjustment_number)].replace(' ', '_')
    else:
        file_name = custom_file_name
    # write to file
    f = open(file_name+".txt","w+")
    for lines in file_output:
        f.write(lines)
    f.close()

    return adjusted_volume


adjustment_function(volume_number, adjustment_number, voi_volume, tiv_volume, beta_value, sex, age, bmi)
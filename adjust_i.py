import argparse
import pickle


"""
this should be called like so:
    >>>python adjust.py

    and then it outputs numbered Volumes to be adjusted

    then prompts you to enter the number associated with the Volume you want adjusted

    then outputs adjustment methods -- also numbered

    then prompts you to enter the number associated with the adjustment method

    then prompts for further information such as raw volumes etc.

    then outputs the adjusted volume using the chosen adjustment method
"""


# store beta values somewhere
# store model fitted model somewhere

beta_values = {0: 0.893836353, 1: 0.653000982, 2: 0.703072538, 3: 0.711833489, 4: 0.466289446, 5: 0.696672306, 6: 0.71295332, 7: 0.639609375}


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

brain_structures = {0:'Total Brain Volume', 1:'Average volume of Thalamus', 2:'Average volume of Caudate', 3:'Average volume of Putamen', 4:'Average volume of Hippocampus', 5:'Average volume of Amygdala', 6:'Average volume of Accumbens', 7:'Average volume of Pallidum'}

print('Possible volumes to adjust')
for i,structures in enumerate(brain_structures.values()):
    print(i, structures)
volume_number = ''
while (not volume_number.isdigit() or int(volume_number) > 6 or int(volume_number) < 0):
    volume_number = input('Please enter the number corresponding to the volume you want adjusted: ')
print('You have chosen ' + brain_structures[int(volume_number)])

print()

adjustment_methods = {0:'proportion adjustment', 1:'power-proportion adjustment', 2:'general linear model adjustment'}

print('Possible adjustment methods')
for i,methods in enumerate(adjustment_methods.values()):
    print(i, methods)
adjustment_number = ''
while (not adjustment_number.isdigit() or int(adjustment_number) > 2 or int(adjustment_number) < 0):
    adjustment_number = input('Please enter the number corresponding to the adjustment method: ')
print('You have chosen ' + adjustment_methods[int(adjustment_number)])

print()

if adjustment_number == '0':
    """
    proportion adjustment
    needs Volume of Interest and Total Intracranial Volume
    """
    voi_volume = ''
    while (not voi_volume.isnumeric()):
        voi_volume = input('Please enter in mL the volume for your chosen brain structure: ')
    
    tiv_volume = ''
    while (not tiv_volume.isnumeric()):
        tiv_volume = input('Please enter in mL the Total Intracranial Volume: ')
    
    adjusted_volume = float(voi_volume) / float(tiv_volume)
    print('The adjusted Volume is: ' + str(adjusted_volume) + 'mL')

    # save to text file
    f = open(brain_structures[int(volume_number)].replace(' ', '_') + "_proportion_adjusted_volume.txt","w+")
    f.write('Volume of Interest: ' + brain_structures[int(volume_number)] + '\n')
    f.write('Volume of Interest Volume: ' + voi_volume + '\n')
    f.write('Total Intracranial Volume: ' + tiv_volume + '\n')
    f.write('The adjusted Volume is: ' + str(adjusted_volume) + 'mL')
    f.close()

if adjustment_number == '1':
    """
    power-proportion adjustment
    needs Volume of Interest and Total Intracranial Volume and fitted beta value
        - fitted beta value can be custom or the pre-fitted value
    """
    voi_volume = ''
    while (not voi_volume.isnumeric()):
        voi_volume = input('Please enter in mL the volume for your chosen brain structure: ')
    
    tiv_volume = ''
    while (not tiv_volume.isnumeric()):
        tiv_volume = input('Please enter in mL the Total Intracranial Volume: ')
    
    beta_value = ''
    custom_beta = ''
    while ( True ):
        custom_beta = input('Use a custom beta value? [y/n] ')
        if custom_beta == "y" or custom_beta == "n":
            break
    if custom_beta == 'y':
        while (not isfloat(beta_value)):
            beta_value = input('Please enter custom beta value: ')

    if custom_beta == 'n':
        beta_values = {0: 0.893836353, 1: 0.653000982, 2: 0.703072538, 3: 0.711833489, 4: 0.466289446, 5: 0.696672306, 6: 0.71295332, 7: 0.639609375}
        beta_value = beta_values[volume_number]

    adjusted_volume = float(voi_volume) / (float(tiv_volume) ** beta_value)
    print()
    print('The adjusted Volume is: ' + str(adjusted_volume) + 'mL')
    # save to text file
    f = open(brain_structures[int(volume_number)].replace(' ', '_') + "_power_proportion_adjusted_volume.txt","w+")
    f.write('Volume of Interest: ' + brain_structures[int(volume_number)] + '\n')
    f.write('Volume of Interest Volume: ' + voi_volume + '\n')
    f.write('Total Intracranial Volume: ' + tiv_volume + '\n')
    f.write('Beta value: ' + str(beta_value) + '\n')
    f.write('Custom beta value?: ' + custom_beta + '\n')
    f.write('The adjusted Volume is: ' + str(adjusted_volume) + 'mL')
    f.close()

if adjustment_number == '2':
    """
    general linear model adjustment
    needs Volume of Interest, Total Intracranial Volume, Sex, Age, BMI 
    """
    voi_volume = ''
    while (not voi_volume.isnumeric()):
        voi_volume = input('Please enter in mL the volume for your chosen brain structure: ')
    
    tiv_volume = ''
    while (not tiv_volume.isnumeric()):
        tiv_volume = input('Please enter in mL the Total Intracranial Volume: ')
    
    sex = ''
    while ( True ):
        sex = input('Please enter sex, 0 for female and 1 for male: ')
        if sex == "1" or sex == "0":
            break

    age = ''
    while (not age.isnumeric()):
        age = input('Please enter in years age: ')

    bmi = ''
    while (not bmi.isnumeric()):
        bmi = input('Please enter in BMI: ')
    
    print()

    brain_structures_ = {'0':'Total_Brain_Volume', '1':'Thalamus', '2':'Caudate', '3':'Putamen', '4':'Hippocampus', '5':'Amygdala', '6':'Accumbens', '7':'Pallidum'}
    # Load model from file
    with open("pickle_model_" + brain_structures_[volume_number] + ".pkl", 'rb') as file:
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
    # save to text file
    f = open(brain_structures[int(volume_number)].replace(' ', '_') + "_glm_adjusted_volume.txt","w+")
    f.write('Volume of Interest: ' + brain_structures[int(volume_number)] + '\n')
    f.write('Volume of Interest Volume: ' + voi_volume + '\n')
    f.write('Total Intracranial Volume: ' + tiv_volume + '\n')
    if sex == "0":
        f.write('Sex: female' + '\n')
    else:
        f.write('Sex: male' + '\n')
    f.write('Age: ' + age + '\n')
    f.write('BMI: ' + bmi + '\n')
    f.write('The adjusted Volume is: ' + str(adjusted_volume[0][0]) + 'mL')
    f.close()


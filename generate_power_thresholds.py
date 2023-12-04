'''
read candidate sigma profile from candidate_sigma_profile.csv
csv is in the format:
sigma, power, z, num_independent_trials
1.011972,25.900000,1,65536
2.010998,29.800000,1,65536
3.000491,35.400000,1,65536
'''

import csv

# prompt user for sigma value
sigma = float(input("Enter target sigma threshold value: "))

# prompt user for z-max value
z_max = 1024

# prompt user for number of independent trials
num_independent_trials = int(input("Enter number of independent trials: "))

whole_csv = []

# read the whole csv file into a list
with open('candidate_sigma_profile.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)  # Skip the header row
    for row in reader:
        whole_csv.append(row)

# extract unique num_independent_trials values into new list
num_independent_trials_values = []
for row in whole_csv:
    num_independent_trials_values.append(row[3])

num_independent_trials_values = list(set(num_independent_trials_values))

# find the nearest num_independent_trials value to the user input
nearest_num_independent_trials = min(num_independent_trials_values, key=lambda x:abs(int(x)-num_independent_trials))

# extract the sigma, power, z values for the nearest num_independent_trials value into new list
nearest_num_independent_trials_values = []
for row in whole_csv:
    if row[3] == nearest_num_independent_trials:
        nearest_num_independent_trials_values.append(row)
        
# for every z value up to zmax
# extract all rows with that z value into new list
# find the power value for the sigma value that is nearest to the user input
# append the sigma, power, and z value to a new list

power_thresholds = []
for z in range(1, z_max + 1):
    z_values = []
    for row in nearest_num_independent_trials_values:
        if row[2] == str(z):
            z_values.append(row)
    nearest_sigma = min(z_values, key=lambda x:abs(float(x[0])-sigma))
    power_thresholds.append([nearest_sigma[0], nearest_sigma[1], nearest_sigma[2]])
    
# write only the power values to a csv file, dont write the sigma or z values i.e. only write the second column
with open('power_thresholds.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in power_thresholds:
        writer.writerow([row[1]])
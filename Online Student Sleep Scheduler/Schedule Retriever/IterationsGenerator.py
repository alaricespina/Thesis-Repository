from pprint import pprint

tiredness_index = [0,1,2,3,4] # 0 is no tiredness, 4 is tired
current_action = ["CLASS", "EAT", "NONE", "REST", "STUDY"]
previous_action = ["CLASS", "EAT", "NONE", "REST", "STUDY"]
next_schedule = ["CLASS", "EAT", "NONE", "REST", "STUDY"]
result_actions = ["CLASS", "EAT", "REST", "STUDY"]

results_array = []
for tiredness in tiredness_index:
    for current in current_action:
        for previous in previous_action:
            for next in next_schedule:
                result = f"{tiredness},{current},{previous},{next},"
                suggest = ""
                
                #If not tired, and free on next sched just study
                not_tired = tiredness < 3 
                FoN = next == "NONE" or "REST"
                if not_tired and FoN:
                    suggest = "STUDY"

                #If eat on next, neat
                if next == "EAT":
                    suggest = "EAT"
               
                #If has class on next, always go to class
                if next == "CLASS":
                    suggest = "CLASS"

                #If Tired, class from previous, and none in next
                tired = tiredness >= 3
                Cp = previous == "CLASS"
                Nn = next == "NONE"
                if tired and Nn and Cp:
                    suggest = "REST"
                
                if tired and next == "REST":
                    suggest = "REST"

                #If tired, and eat or none from previous with none as next
                tired = tiredness >= 3 
                EoNfp = previous == "EAT" or previous == "NONE"
                Nn = next == "NONE"
                if tired and Nn and EoNfp:
                    suggest = "STUDY"

                loaded = (current == "CLASS" or current == "STUDY") and (next=="CLASS" or next == "STUDY" or next=="NONE")
                if tired and loaded:
                    suggest = "REST"

                breaked = (current == "CLASS" or current == "STUDY") and (next == "NONE" or next == "STUDY")
                if tired and breaked:
                    suggest = "STUDY"
                
                not_loaded = (current == "EAT" or current == "NONE" or current == "REST") and (next == "NONE" or next == "STUDY")
                if tired and not_loaded: 
                    suggest = "STUDY"

                result += f"{suggest}\n"  
                results_array.append(result)

pprint(results_array)

def write_file():
    file_name = "Iterations.csv"
    file = open(file_name, "w")
    file.writelines(results_array)
    file.close()

write_file()
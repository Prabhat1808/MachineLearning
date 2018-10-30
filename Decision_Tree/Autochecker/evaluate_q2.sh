#!/bin/bash
# RUN: ./autograde_q2.sh users.txt data sandbox submissions
run()
{
    timelimit="$1"
    shift
    logs="$1"
    shift
    chmod +x "$1"

    #echo -e "\nEvaluating Part $2"
    echo -e "\nEvaluating Part $2" >> "$logs"
    
    time_start=$(date +%s)
    timeout -k "$timelimit" "$timelimit" "./$@" &>> "$logs"
    status=$?
    time_end=$(date +%s)
    
    user_time=$(( time_end - time_start ))

    if [ "$status" -eq 124 ]; then
        echo "Status: Timed out for Part $2"
        echo "Status: Timed out Part $2" >> "$logs"
    else
        echo -e "Finished running part $2 "
        echo "Status: OK, Time taken: $user_time"
        echo "Status: OK, Time taken: $user_time" >> "$logs"
        
    fi

    write_score ",$timelimit,$user_time"
}

compute_score()
{
    
    # Compute score as per predicted values and write to given file
    # $1 python_file
    # $2 targets
    # $3 predicted
    # $4 outfile
    if [ -f "$3" ]; then
        score=$(python3 "$1" "$2" "$3" "$4")
        echo -e "F-Score on dummy target values is $score ."
        echo -e "Upload your predictions on Server to get the actual accuracy on public dataset."
    else
        echo -e "\nPrediction file not found at $3 .Please correct it."
        score="0.0"
    fi
    write_score ",$score"
}

write_score()
{
    echo -n "$1" >> "$score_file"
}

evaluate()
{
    # $1: data_dir
    # $2: sandbox_dir
    # $3: entry_number
    # $4: submissions_dir 

    main_dir=$(pwd)
    
    if ! [ -d "$2" ]; then
        mkdir -p "$2"
    fi
    
    entry_number="$3"

    stud_folder_path=$(realpath "$2/$3")
    logs=$(realpath "$2")"/logs"
    score=$(realpath "$2")"/score"
    

    if  [ -d "$stud_folder_path" ]; then
        echo "Deleting old folder in sandbox"
        rm -r "$stud_folder_path"
    fi

    if [ -f "${4}/${3}.zip" ]; then
        echo -e "\n${3}.zip Found"
        echo -e "Extracting ${3}.zip"
        unzip -qq "$4/$3".zip -d "$2"
    elif [ -f "${4}/${3}.rar" ]; then
        echo -e "\n${3}.rar Found. You should change rar to zip to avoid penalty"
        echo -e "Extracting ${3}.rar"
        unrar x "$4/$3.rar" "$2"
    fi    

    if ! [ -d "$logs" ]; then
        mkdir -p "$logs"
    fi
    if ! [ -d "$score" ]; then
        mkdir -p "$score"
    fi
    
    logs="$logs/log.txt"
    score_file="$score/scores.txt"
    data_folder_path=$(realpath "$1")
    compute_accuracy=$(realpath "compute_fscore.py")
    cd "$stud_folder_path"    
    
    status="OK"
    if [ -f "dtree" ]; then
        fname="dtree" 
        bash_fname_penalty="NO"
        dos2unix dtree
    elif [ -f "dtree.sh" ]; then
        fname="dtree.sh"
        echo -e " .sh file found. You should remove sh it to avoid penalty"
        bash_fname_penalty="YES_sh"
        dos2unix dtree.sh
    else
        status="NA"
        #status="OK"
        bash_fname_penalty="YES"
        echo -e "Bash file name incorrect/not found.Follow instructions"
    fi

    write_score "$entry_number"
    write_score ",$bash_fname_penalty"

    time_a="3600"
    time_b="3600"    
    
    if [ $status == "OK" ]; then

        part="a"
        echo -e "\nEvaluating Decision Tree part(a)"
        run "$time_a" "$logs" "$fname" "$part" "$data_folder_path/train.csv" "$data_folder_path/valid.csv" "$data_folder_path/test.csv" "$stud_folder_path/predictions_dtree_${part}"  "$stud_folder_path/${entry_number}_${part}.png" 
        compute_score "$compute_accuracy" "$data_folder_path/dummy_dtree_target_labels.txt" "$stud_folder_path/predictions_dtree_${part}.txt" "$stud_folder_path/result_dtree_${part}.txt"           

        part="b"
        echo -e "\nEvaluating Decision Tree part(b)"
        run "$time_b" "$logs" "$fname" "$part" "$data_folder_path/train.csv" "$data_folder_path/valid.csv" "$data_folder_path/test.csv" "$stud_folder_path/predictions_dtree_${part}" "$stud_folder_path/${entry_number}_${part}.png" 	
        compute_score "$compute_accuracy" "$data_folder_path/dummy_dtree_target_labels.txt" "$stud_folder_path/predictions_dtree_${part}.txt" "$stud_folder_path/result_dtree_${part}.txt" 
        
        
        echo -n $'\n' >> "$score_file"

    else        
        write_score ",$t1,NA,0.0,$t2,NA,0.0,$t3,NA,0.0"
    fi
    cd "$main_dir"
}

infile="$1"
data_dir="$2"
sandbox_dir="$3"
submissions_dir="$4"
#write_score "ENTRY_NUM,PEN_BASH,TIME_a,FScore_a,TIME_b,FScore_b"

while read entry_num;do     
      echo -e "\nEvaluating Entry No $entry_num"  
	evaluate "$data_dir" "$sandbox_dir" "$entry_num" "$submissions_dir" 
      echo -e "--------------------------------------------------------"
      echo -e "--------------------------------------------------------"
done < $infile

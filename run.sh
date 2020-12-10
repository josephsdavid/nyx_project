#!/usr/bin/env bash

no_wandb='false'
hup='true'

file=${!#}

logfile=$(echo "$file" | awk -F"." '{print $1}'  | awk -F"/" '{print $2}')

run_file() {
	if [[ "$file" =~ \.yaml ]]; then
		sweep=$(wandb sweep "$file" 2>&1 >/dev/null | tail -n 1 | cut -c 42-)
		echo "$sweep"
		if [ "$hup" = "false" ]; then
			wandb agent "$sweep" &
		else
			wandb agent "$sweep"
		fi
	else
		bash "$file"
	fi
}

print_usage(){
	declare -a help=(
	"Wandb Runner Script"
	"Usage:"
	"./run.sh -[nl] filename"
	"If the file is a bash script, will simply run the script. If the file is a yaml file, it will start the sweep and run wandb agent on the sweep"
       	"Flags:"
	"-n: dont use wandb logging, calls 'wandb off' before the file is run, then after completion or failure, calls 'wandb on'"
       	"-l: logoff, output redirected to nohup.out, program not stopped on disconnect. To view progress, run 'tail -f nohup.out'"
	)

	printf "%s\n%s\n\n\t%s\n\t%s\n\n%s\n\n\t%s\n\t%s\n\t%s\n%c" "${help[@]}" # this is horrible

}

if [[ $# -eq 0 ]] ; then
	print_usage
	exit 0
fi

while getopts 'nl' flag ; do
	case "${flag}" in
		n) no_wandb='true' ;;
		l) hup='false' ;;
		*) print_usage
			exit 1 ;;
	esac
done

if [ "$no_wandb" = true ] ; then
	wandb off
fi

if [ "$hup" = false ] ; then
	trap '' HUP
	echo "redirecting output to ${logfile}.log..."
	run_file >> "${logfile}.log" 2>&1 &
else
	run_file
fi



if [ "$no_wandb" = true ] ; then
	wandb on
fi

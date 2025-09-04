{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --gres=gpu
#SBATCH -p gpu,seas_gpu
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --job-name="{{ id }}"
{% if partition %}
#SBATCH --partition={{ partition }}
{% endif %}
{% set walltime = operations | calc_walltime(parallel) %}
        {% if walltime %}
#SBATCH -t {{ walltime|format_timedelta }}
        {% endif %}
{% if job_output %}
#SBATCH --output={{ job_output }}
#SBATCH --error={{ job_output }}

{% endif %}
{% block tasks %}
{% endblock %}
{% endblock %}
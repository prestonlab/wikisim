#!/bin/bash
#
# Create individual masks from a group cluster.

if [[ $# -lt 4 ]]; then
    cat << EOF
bender_clust_mask.sh - Create individual masks from a group cluster

Usage: bender_clust_mask.sh [-r radius] [-i intersect] subjects cluster_mask clustind maskname

-r radius
    radius for dilating individual-subject masks

-i intersect
    intersect individual masks with this mask

subjects
    colon-separated list of subject numbers to include

cluster_mask
    image with labeled clusters

clustind
    index of cluster to use for mask

maskname
    name of mask to create

EOF
    exit 1
fi

radius=0
intersect=""
while getopts ":r:i:" opt; do
    case $opt in
    r)
        radius=$OPTARG
        ;;
    i)
        intersect=$OPTARG
        ;;
    *)
        echo "Invalid option: $opt"
        exit 1
    esac
done
shift $((OPTIND-1))

subjects=$1
cluster_mask=$2
clustind=$3
maskname=$4

if [[ $(imtest "$cluster_mask") = 0 ]]; then
    echo "cluster mask does not exist: $cluster_mask"
    exit 1
fi

parent=$(dirname "$cluster_mask")

# make group-level mask (saved to same directory as the cluster mask)
echo "Making group mask..."
group_mask=$parent/${maskname}.nii.gz
fslmaths "$cluster_mask" -thr "$clustind" -uthr "$clustind" -bin "$group_mask"
n_vox=$(fslstats "$group_mask" -V | cut -d ' ' -f 1)
echo "$maskname: $n_vox voxels"

# make individual masks
echo "Transforming to native spaces..."
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
parallel -P 6 transform_mni2func.sh -a 2 -n NearestNeighbor "$group_mask" \
    "$STUDYDIR/{}/anatomy/bbreg/data/${maskname}.nii.gz" {} \
    ::: $(subjids -s " " "$subjects")

# expand the mask
if (( $(bc <<< "$radius > 0") )); then
    echo "Expanding native masks using a radius of $radius..."
    parallel -P 12 fslmaths \
        "$STUDYDIR/{}/anatomy/bbreg/data/${maskname}.nii.gz" \
        -kernel sphere "$radius" -dilD \
        "$STUDYDIR/{}/anatomy/bbreg/data/${maskname}.nii.gz" \
        ::: $(subjids -s " " $subjects)
fi

# intersect with an individual subject mask
if [[ -n $intersect ]]; then
    echo "Intersecting with $intersect mask..."
    parallel -P 12 fslmaths \
        "$STUDYDIR/{}/anatomy/bbreg/data/${maskname}.nii.gz" \
        -mas "$STUDYDIR/{}/anatomy/bbreg/data/${intersect}.nii.gz" \
        "$STUDYDIR/{}/anatomy/bbreg/data/${maskname}.nii.gz" \
        ::: $(subjids -s " " $subjects)
fi

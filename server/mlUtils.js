module.exports.calculate_metrics = (cMat) => {
    const dim = cMat.length;
    var tAcc  = 0;
    var tPres = 0;
    var tRec  = 0;
    var tF1   = 0;

    for (let c = 0; c < dim; ++c) {
        const metrics = calculate_metric(cMat, c);
        tAcc  += metrics[0];
        tPres += metrics[1];
        tRec  += metrics[2];
        tF1   += metrics[3];
    }

    return [tAcc/dim, tPres/dim, tRec/dim, tF1/dim];

}

// calculates the metrics for a particular class
function calculate_metric(cMat, c) {
    var tp = 0;
    var tn = 0;
    var fp = 0;
    var fn = 0;

    for (let j = 0; j < cMat.length; ++j) {
        for (let i = 0; i < cMat[0].length; ++i) {
            if (i == j && i == c) {
                tp += cMat[i][j];
            } else if (i == j && i != c) {
                tn += cMat[i][j];
            } else if (i == c && j != c) {
                fn += cMat[i][j];
            } else if (j == c && i != c) {
                fp += cMat[i][j];
            }

        }
    }


    var acc = (tp + tn) / (tp + tn + fp + fn);
    var precision = tp / (tp + fp);
    var recall = tp / (tp + fn);
    var f1 = (2 * precision * recall) / (precision + recall);

    if ((tp + tn + fp + fn) == 0){
        acc = 0;
    }
    if ((tp + fp) == 0) {
        precision = 0;
    }
    if ((tp + fn) == 0) {
        recall = 0;
    }
    if ((precision + recall) == 0) {
        f1 = 0;
    }

    return [acc, precision, recall, f1];
}
'''
@author Tian Shi
Please contact tshi@vt.edu
Based on Lin Zhouhan(@hantek) visualization codes.
'''
import os
import random
from codecs import open

import numpy
import scipy


def createHTML(texts, weights1, weights2, rates, pdrates, fileName):
    """
    Creates a html file with text heat.
    weights: attention weights for visualizing
    texts: text on which attention weights are to be visualized
    """
    fOut = open(fileName, "w", encoding="utf-8")
    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body style="width: 400px">
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = """
    var color = ["255,0,0", "0,255,0", "0,255,255", "255,165,0"];
    var ngram_length = 3;
    var half_ngram = 1;
        
    inttt = []
    for (var k=0; k < any_text.length; k++) {
        var tokens = any_text[k].split(" ");
        var intensity = new Array(tokens.length);
        var max_intensity = Number.MIN_SAFE_INTEGER;
        var min_intensity = Number.MAX_SAFE_INTEGER;
        for (var i = 0; i < intensity.length; i++) {
            intensity[i] = 0.0;
            for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
                if (i+j < intensity.length && i+j > -1) {
                    intensity[i] += trigram_weights1[k][i + j];
                }
            }
            if (i == 0 || i == intensity.length-1) {
                intensity[i] /= 2.0;
            } else {
                intensity[i] /= 3.0;
            }
            if (intensity[i] > max_intensity) {
                max_intensity = intensity[i];
            }
            if (intensity[i] < min_intensity) {
                min_intensity = intensity[i];
            }
        }
        var denominator = max_intensity - min_intensity;
        
        for (var i = 0; i < intensity.length; i++) {
            intensity[i] = (intensity[i] - min_intensity) / denominator;
        }
        inttt.push(intensity)
        
    }
    var example = ['Feel', 'Look', 'Smell', 'Taste']
    var heat_text  = "<p style='margin: 0; padding: 5px;'><b>";
    ss = ", ";
    for (var i = 0; i < any_text.length; i++){
        heat_text += "<span style='color:rgba(" + color[i] + ", 0.7)'>" + example[i] + "</span>";
        heat_text += ": (" + rate[i] + "," + pdrate[i] + ")" + ss;
        if (i == any_text.length - 2) {
            ss = " ";
        }
    }
    heat_text += "</b><br>"
    heat_text += "<p style='border: 1px solid black; margin: 0; padding: 5px;'>"
    var ddd = new Array(tokens.length);
    var iii = new Array(tokens.length);
    for (var i = 0; i < intensity.length; i++) {
        ddd[i] = 0.0
        iii[i] = -1
        for (var k = 0; k < any_text.length; k++ ){
            if (inttt[k][i] > ddd[i]) {
                ddd[i] = inttt[k][i]
                iii[i] = k
            }
        }
    }
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
        heat_text += "<span style='background-color:rgba(" + color[iii[i]] + "," + ddd[i] + ")'>" + space + tokens[i] + "</span>";
        if (space == "") {
            space = " ";
        }
    }
    
    
    inttt = []
    for (var k=0; k < any_text.length; k++) {
        var tokens = any_text[k].split(" ");
        var intensity = new Array(tokens.length);
        var max_intensity = Number.MIN_SAFE_INTEGER;
        var min_intensity = Number.MAX_SAFE_INTEGER;
        for (var i = 0; i < intensity.length; i++) {
            intensity[i] = 0.0;
            for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
                if (i+j < intensity.length && i+j > -1) {
                    intensity[i] += trigram_weights2[k][i + j];
                }
            }
            if (i == 0 || i == intensity.length-1) {
                intensity[i] /= 2.0;
            } else {
                intensity[i] /= 3.0;
            }
            if (intensity[i] > max_intensity) {
                max_intensity = intensity[i];
            }
            if (intensity[i] < min_intensity) {
                min_intensity = intensity[i];
            }
        }
        var denominator = max_intensity - min_intensity;
        
        for (var i = 0; i < intensity.length; i++) {
            intensity[i] = (intensity[i] - min_intensity) / denominator;
        }
        inttt.push(intensity)
        
    }
    
    heat_text += "<p style='border: 1px solid black; margin: 0; padding: 5px;'>"
    var ddd = new Array(tokens.length);
    var iii = new Array(tokens.length);
    for (var i = 0; i < intensity.length; i++) {
        ddd[i] = 0.0
        iii[i] = -1
        for (var k = 0; k < any_text.length; k++ ){
            if (inttt[k][i] > ddd[i]) {
                ddd[i] = inttt[k][i]
                iii[i] = k
            }
        }
    }
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
        heat_text += "<span style='background-color:rgba(" + color[iii[i]] + "," + ddd[i] + ")'>" + space + tokens[i] + "</span>";
        if (space == "") {
            space = " ";
        }
    }
    document.body.innerHTML += heat_text;
    
    </script>
    </html>"""
    def putQuote(x): return "\"%s\"" % x
    textsString = "var any_text = [%s];\n" % (",".join(map(putQuote, texts)))
    weightsString1 = "var trigram_weights1 = [%s];\n" % (
        ",".join(map(str, weights1)))
    weightsString2 = "var trigram_weights2 = [%s];\n" % (
        ",".join(map(str, weights2)))
    rateString = "var rate = [%s];\n" % (",".join(map(str, rates)))
    pdrateString = "var pdrate = [%s];\n" % (",".join(map(str, pdrates)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString1)
    fOut.write(weightsString2)
    fOut.write(rateString)
    fOut.write(pdrateString)
    fOut.write(part2)
    fOut.close()

    return

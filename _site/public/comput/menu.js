
var menuoptions =
[
  ["Intro, Chemical Kinetics", "index.html", "---"],
  ["Book Introduction", "index.html"],
  ["Exponential Decay", "decay.html"],
  ["Channel Gating","channels.html"],
  ["Hill Function", "empty.html"],//"hill.html"],
  ["Recap & Further Reading", "empty.html"],//"kinetics_recap.html"],
  ["Exercises", "empty.html"],//"kinetics_exercises.html"],
  ["Conductance-Based Models", "empty.html", "---"],
  ["Voltage-Dependent Gating", "empty.html"],
  ["Membrane Potential", "empty.html"],
  ["Passive Neuron Model", "empty.html"],
  ["Hodgkin-Huxley Model", "empty.html"],
  ["General Recipe", "empty.html"],
  ["Exercises", "empty.html"],
  ["Multi-Compartment Models", "empty.html", "---"],
  ["Cable Theory", "empty.html"],
  ["Discretizing Morphologies", "empty.html"],
  ["Passive Membranes", "empty.html"],
  ["Active Membranes", "empty.html"],
  ["Challenges", "empty.html"],
  ["Exercises", "empty.html"],
  ["Phase Plane Analysis", "empty.html", "---"],
  ["Stability and Instability", "empty.html"],
  ["Bistability in 1D", "empty.html"],
  ["Oscillations in 2D", "empty.html"],
  ["Bifurcations", "empty.html"],
  ["Type I, II, III excitability", "empty.html"],
  ["Integrators and Resonators", "empty.html"],
  ["Exercises", "empty.html"],
  ["Appendices", "appendices.html", "---"],
  ["Math Primer", "empty.html"],//"math_appendix.html"],
  ["Programming Primer", "empty.html"],//"programming_appendix.html"],
];


var currentIndex;
var currenttd;

for (var i = 0; i < menuoptions.length; i++)
{
  var ispartheader = menuoptions[i].length === 3;
  var pathArray = location.pathname.substring(1).split("/");
  var currentPage = pathArray[pathArray.length - 1];
  var className = currentPage === menuoptions[i][1] && i != 0
    ? 'sectionButton current'
    : 'sectionButton';

  var style = currentPage === menuoptions[i][1] && !ispartheader && i != 0
    ? "color: #bbb;"
    : "";

  style += ispartheader
    ? "font-size: 9.5px; color: rgb(200,200,200); font-weight:bold; opacity: 1.0"
    : "";

  if (ispartheader)
  {
    currenttd = d3.select("#menurow").append("td")
      .attr('class', "partdata");
  }

  if (currentPage === menuoptions[i][1])
  {
    currentIndex = i;
  }

  currenttd.append("div")
    .attr('class', className)
      .append("a")
      .attr('href', menuoptions[i][1])
      .attr("style", style)
      .text(menuoptions[i][0].toUpperCase());
    //.text(menuoptions[i].toUpperCase());
}


var titleContents =
'<div style="color: #ddd; font-size: 22px">MODELING <span style="color:rgb(93,217,241)">CHANNELS</span>, <span style="color:rgb(243,48,110);">SYNAPSES</span>, AND <span style="color:rgb(232,218,106);">SPIKES</span><!--<svg id="phasortitle" class="svgWithText" width="500" height="20" style="display: inline">--></svg></div><div class="subheader" style="margin-top: 0px; color: #888; font-size: 18px; width: 800px">A SHORT AND SIMPLE INTRODUCTION TO THEORETICAL NEUROSCIENCE<span id="icons" style="margin-left: 10"></span></div>';

document.getElementById('titleinfo').innerHTML = titleContents;

document.getElementById('icons').innerHTML = 
'<span class="icon-twitter bigicon" style="color: #aaa; cursor: pointer;" onclick="window.open(\'http://www.twitter.com/jackschaedler\');"> <span style="font-size:10px; color: #555"></span></span><span class="icon-github-circled bigicon" style="color: #aaa; cursor: pointer" onclick="window.open(\'http://www.github.com/jackschaedler/circles-sines-signals\');"><span style="font-size:10px; padding-left: 4px; color: #555"></span></span>';

if (document.getElementById('footer') != null)
{
  document.getElementById('footer').innerHTML = '<div class="footerbutton" style=""><a href="' + menuoptions[currentIndex + 1][1] + '">NEXT <span style="font-size: 12px">▶</span></a></div>';
}
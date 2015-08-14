var EXP_DERIV = (function() {
	//// Set up SVG window, related parameters, and helper functions
	var w = 350;
	var h = 300;
	var lr_padding = 175;
	var ud_padding = 40;
	
	function exp_func(x){
		return Math.exp(-RATE_CONST*x);
	}

	//Create scale functions
	var xScale = d3.scale.linear()
						.domain([0, 3])
						.range([lr_padding,w+lr_padding]);

	var yScale = d3.scale.linear()
						.domain([0, 1.3])
						.range([h - ud_padding, 0]);
	

	//Create SVG element to contain animation
	var svg = d3.select("#exp_deriv_svg");

	//Exponential decay line
	function get_exp_line(){
		var lineData = []
		for (xx = 0; xx < 3; xx+=0.01) { 
	    	lineData.push({ "x": xx, "y": exp_func(xx)});
		}
		return lineData;
	}
	

	//line accessor function
	var lineFunction = d3.svg.line()
	                      .x(function(d) { return xScale(d.x); })
	                      .y(function(d) { return yScale(d.y); })
	                     .interpolate("linear");


	//The line SVG Path we draw
	var lineGraph = svg.append("path")
	                            .attr("d", lineFunction(get_exp_line()))
	                            .attr("stroke", "red")
	                            .attr("stroke-width", 3)
	                            .attr("fill", "none");
	

	// X axis and label
	var xAxis = d3.svg.axis()
						.scale(xScale)
						.orient("bottom")
						.ticks(5); 
	svg.append("g")
		.attr("class", "x axis")
		.attr("transform", "translate(0," + yScale.range()[0] + ")")
		.call(xAxis);
	svg.append("text")
		.attr("class", "x label")
		.attr("text-anchor", "middle")
		.attr("dominant-baseline", "text-after-edge")
		.attr("x", (w + 2*lr_padding)/2 )
		.attr("y", h)
		.attr("font-weight", "bold")
		.attr("font-size","12px")
		.text("time (s)");
	
	// Y axis and label
	var yAxis = d3.svg.axis()
					.scale(yScale)
					.orient("left")
					.ticks(5);
	svg.append("g")
            .attr("id", "yy")
            .attr("class", "axis")
            .attr("transform", "translate(" + lr_padding + ",0)")
            .call(yAxis);
	svg.append("text")
		.attr("class", "y label")
		.attr("text-anchor", "middle")
		.attr("y", lr_padding)
		.attr("x", -h/2)
		.attr("dy", "-35px")
		.attr("transform", "rotate(-90)")
		.attr("font-weight", "bold")
		.attr("font-size","12px")
		.text("Concentration [X]");

    // Moving tangent line
    var x0 = 0.5;
    var y0 = exp_func(x0);
    var m = -y0;
	var tanData = [{ "x": 0, "y": -m*x0+y0},{ "x": (-y0/m)+x0, "y": 0}];

	//The line SVG Path we draw
	var tanlineGraph = svg.append("path")
	                            .attr("d", lineFunction(tanData))
	                            .attr("stroke", "blue")
	                            .attr("stroke-width", 3)
	                            .attr("fill", "none");

    svg.append("rect")
      .attr("class", "svg_overlay")
      .attr("width", w)
      .attr("height", h - ud_padding)
      .attr("transform","translate(" + lr_padding + "," + ud_padding + ")")
      .on("mouseover", function() { focus.style("display", null); })
      .on("mousemove", mousemove);

    // Moving circle and text
	var focus = svg.append("g")
      .attr("class", "focus");

  	focus.append("circle")
      .attr("r", 4.5);

  	focus.append("text")
      .attr("x", 9)
      .attr("dy", "-5px")
      .attr("font-weight","bold")
      .attr("font-size","13px");
	focus.attr("transform", "translate(" + xScale(x0) + "," + yScale(y0) + ")");
    focus.select("text").text("[X] = "+d3.round(y0,2));

    rate_text = svg.append("text")
		.attr("class", "x label")
		.attr("text-anchor", "middle")
		.attr("x", (w + 2*lr_padding)/2 )
		.attr("y", ud_padding)
		.attr("font-weight", "bold")
		.attr("font-size","14px")
		.attr("fill","red")
		.text("Rate Constant (k): "+RATE_CONST);

	tan_text = svg.append("text")
		.attr("class", "x label")
		.attr("text-anchor", "middle")
		.attr("x", (w + 2*lr_padding)/2 )
		.attr("y", ud_padding+20)
		.attr("font-weight", "bold")
		.attr("font-size","14px")
		.attr("fill","blue")
		.text("Slope of Tangent Line = -k [X] = "+m);

	function mousemove() {
	    x0 = xScale.invert(d3.mouse(this)[0]+lr_padding);
	}

	function redraw(){
		y0 = exp_func(x0)
	    m = -RATE_CONST*y0;
	    if((-y0/m)+x0 < 3) {
	    	tanData = [{ "x": 0, "y": -m*x0+y0},{ "x": (-y0/m)+x0, "y": 0}];
	    } else {
	    	tanData = [{ "x": 0, "y": -m*x0+y0},{ "x": 3, "y":m*(3-x0)+y0}]; // prevent line from extending off xrange
	    }
	    tanlineGraph.transition().attr("d", lineFunction(tanData)).duration(0);
	    lineGraph.transition().attr("d", lineFunction(get_exp_line())).duration(0);
	    rate_text.transition().text("Rate Constant (k) = "+RATE_CONST).duration(0);
	    tan_text.transition().text("Slope = -k [X] = "+d3.round(m,2)).duration(0);
	    focus.attr("transform", "translate(" + xScale(x0) + "," + yScale(y0) + ")");
	    focus.select("text").text("[X] = "+d3.round(y0,2));
	}

	d3.timer(redraw,10);

})();

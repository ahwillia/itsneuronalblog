(function() {
	//// Set up SVG window, related parameters, and helper functions
	var w = 740;
	var h = 300;
	var lr_padding = 130;
	var ud_padding = 10;
	
	//Create scale functions
	var xScale = d3.scale.linear()
						.domain([-1, 1])
						.range([lr_padding, w - lr_padding * 2]);

	var yScale = d3.scale.linear()
						.domain([-1, 1])
						.range([h - ud_padding, ud_padding]);


	//Create SVG element to contain animation
	var svg = d3.select("#single_protein_degration_svg");
	svg.selectAll("circle")
		.data([{x:0, y:0, r:(h/2)-ud_padding }])
		.enter()
		.append("circle")
		.attr("cx", function(d) {  return xScale(d.x);  } )
		.attr("cy", function(d) {  return yScale(d.y);  }  )
		.attr("r", function(d) {  return d.r;  } )
		.style("fill","white")
		.style("stroke","black")
		.style("stroke-width",4.0);

	// Function to setInterval with zero delay for "totalTime" milliseconds
	function setIntervalX(callback, totalTime) {
		var startDate = new Date();
		var curDate = null;
		var intervalID = window.setInterval(function () {
			callback();
			curDate = new Date();
			if ((curDate-startDate) > totalTime) {
				window.clearInterval(intervalID);
				displayResults(totalTime/1000);
			}
		}, 0);
	}

	//// ANIMATION CODE

	// Brownian motion function (with gravity towards center and friction)
	function motion(e, index, array) {
		e.xloc = e.xloc + e.xvel;
		e.yloc = e.yloc + e.yvel;
		e.xvel = e.xvel + 0.004*(Math.random()-0.5) - 0.05*e.xvel - 0.00001*e.xloc;
		e.yvel = e.yvel + 0.004*(Math.random()-0.5) - 0.05*e.yvel - 0.00001*e.yloc;
		if(e.xloc > 1) { e.xloc = 1; }
		if(e.xloc < -1) { e.xloc = -1; }
		if(e.yloc > 1) { e.yloc = 1; }
		if(e.yloc < -1) { e.yloc = -1; }
	}

	// Function to update brownian motion for all circles
	function update(){
		pos.forEach(motion);
		svg.selectAll("#protein")
			.data(pos)
			.attr("cx", function(d) { return xScale(d.xloc); })
			.attr("cy", function(d) { return yScale(d.yloc); });
	}

	function runSimulation() {
		svg.selectAll("#protein").data([]).exit().remove();
		svg.selectAll("text").data([]).exit().remove();
	
		// Calculate time to degradation
		var q = 0.5;                 // probability of degrading in one second
		var lambda = -Math.log(1-q); // rate parameter of exp distribution
		// sample from inverse CDF for time to degradation [sec]
		var t_end = d3.round(-Math.log(1-Math.random()) / lambda,3);
	
		// Initial position
		pos = [ {xloc: 0, yloc: 0, xvel: 0, yvel: 0} ];
	
		// Create circles
		svg.selectAll("#protein")
			.data(pos)
			.enter()
			.append("circle")
			.attr("cx", function(d) {  return xScale(d.xloc);  })
			.attr("cy", function(d) {  return yScale(d.yloc);  })
			.attr("r", 6 )
			.attr("id","protein");
	 
		setIntervalX(update, t_end*1000);
	}

	function displayResults(t_end) {
		svg.selectAll("#protein").data([]).exit().remove();
	 
		// display t_end
		svg.selectAll("text")
			.data([{x:0,y:0}])
			.enter()
			.append("text")
			.text(function(d) {return "Time Elapsed = " + t_end + " seconds"} )
			.attr("x", function(d) {return xScale(d.x)} )
			.attr("y", function(d) {return yScale(d.y)} )
			.style("text-anchor","middle")
			.style("fill","red")
			.style("font-size","15px");

		// Automatically run next trial after delay
		setTimeout(runSimulation, 2500);
	}

	runSimulation();

})();

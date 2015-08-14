(function() {
	//// Set up SVG window, related parameters, and helper functions
	var w = 400;
	var h = 300;
	var lr_padding = 20;
	var mid_padding = 75;
	var ud_padding = 40;
	var w_hist = 350;
	
	var nProteins = 250;
	var q = 0.5;                 // probability of degrading in one second
	var lambda = -Math.log(1-q); // rate parameter of exp distribution

	//Create scale functions
	var xScale = d3.scale.linear()
						.domain([-1, 1])
						.range([lr_padding,w-mid_padding]);

	var yScale = d3.scale.linear()
						.domain([-1, 1])
						.range([h - ud_padding, ud_padding]);
	
	var xScale_hist = d3.scale.linear()
						.domain([0, 10])
						.range([w, w + w_hist - lr_padding]);
						
	var yScale_hist = d3.scale.linear()
						.domain([0, 100])
						.range([h - ud_padding, ud_padding]);
	

	//Create SVG element to contain animation
	var svg = d3.select("#many_protein_degration_svg");
	svg.selectAll("#box")
		.data([{x:-1, y:1 }])
		.enter()
		.append("rect")
		.attr("x", function(d) {  return xScale(d.x);  } )
		.attr("y", function(d) {  return yScale(d.y);  }  )
		.attr("width", w - mid_padding - lr_padding)
		.attr("height", h - ud_padding*2)
		.style("fill","white")
		.style("stroke","black")
		.style("stroke-width",4.0);
	
	// Add histogram bars with no height
	var hist = d3.layout.histogram()
				.bins(xScale_hist.ticks(20))
				([]);
	
	var bar = svg.selectAll(".bar")
				.data(hist)
				.enter().append("g")
				.attr("class", "bar");
				
	bar.append("rect")
				.attr("id","histrect")
				.attr("x", function(d) { return xScale_hist(d.x); } )
				.attr("y", function(d) { return yScale_hist(0); })
				.attr("width", xScale_hist(hist[0].dx)-xScale_hist(0))
				.attr("height", 0);
	
	// X axis and label for histogram
	var xAxis = d3.svg.axis()
						.scale(xScale_hist)
						.orient("bottom")
						.ticks(5); 
	svg.append("g")
		.attr("class", "x axis")
		.attr("transform", "translate(0," + yScale_hist.range()[0] + ")")
		.call(xAxis);
	svg.append("text")
		.attr("class", "x label")
		.attr("text-anchor", "middle")
		.attr("dominant-baseline", "text-after-edge")
		.attr("x", w + (w_hist - lr_padding)/2 )
		.attr("y", h)
		.attr("font-weight", "bold")
		.attr("font-size","12px")
		.text("time (s)");
	
	// Y axis and label for histogram
	var yAxis = d3.svg.axis()
					.scale(yScale_hist)
					.orient("left")
					.ticks(5);
	svg.append("g")
            .attr("id", "yy")
            .attr("class", "axis")
            .attr("transform", "translate(" + w + ",0)")
            .call(yAxis);
	svg.append("text")
		.attr("class", "y label")
		.attr("text-anchor", "middle")
		.attr("y", w)
		.attr("x", -h/2)
		.attr("dy", "-35px")
		.attr("transform", "rotate(-90)")
		.attr("font-weight", "bold")
		.attr("font-size","12px")
		.text("# of degraded proteins");
	
	// Function that updates histogram for each new simulation
	function makeHist(values) {
		hist = d3.layout.histogram()
					.bins(xScale_hist.ticks(20))
					(values);
		
		// first set all heights to zero
		svg.selectAll("#histrect")
					.attr("height", 0)
					.attr("y", function(d) { return yScale_hist(0); });
		
		// then dynamically add hist bars
		svg.selectAll("#histrect")
					.data(hist)
					.transition()
					.duration(500)
					.delay(function(d) { return(d.x*1000); })
					.attr("height", function(d) { return yScale_hist(0)-yScale_hist(d.y); })
					.attr("y", function(d) { return yScale_hist(d.y); });

    }

	//// ANIMATION CODE
	
	// Brownian motion function (with gravity towards center and friction)
	function motion(e, index, array) {
		e.xloc = e.xloc + e.xvel;
		e.yloc = e.yloc + e.yvel;
		e.xvel = e.xvel + 0.004*(Math.random()-0.5) - 0.05*e.xvel - 0.000005*e.xloc;
		e.yvel = e.yvel + 0.004*(Math.random()-0.5) - 0.05*e.yvel - 0.000005*e.yloc;
		if(e.xloc > 1) { e.xloc = 1; }
		if(e.xloc < -1) { e.xloc = -1; }
		if(e.yloc > 1) { e.yloc = 1; }
		if(e.yloc < -1) { e.yloc = -1; }
	}

	// Function to update brownian motion for all circles
	function update(pos){
		pos.forEach(motion);
		svg.selectAll("#protein")
			.data(pos)
			.attr("cx", function(d) { return xScale(d.xloc); })
			.attr("cy", function(d) { return yScale(d.yloc); });
	}

	function runSimulation() {
		svg.selectAll("#protein").data([]).exit().remove();
		svg.selectAll("#result_text").data([]).exit().remove();
	
		// Calculate time to degradation
		
		
		// Initial position and degradation times
		var pos = [];
		var t_end = [];
		for (var i=0; i<nProteins; i++)
		{
			pos.push({xloc: 2*Math.random()-1, yloc: 2*Math.random()-1, xvel: 0, yvel: 0});
			t_end.push(d3.round(-Math.log(1-Math.random()) / lambda,3)); // sample from inverse CDF
		}
		t_end.sort(function(a,b){return a-b}); // sort t_end ascending
		makeHist(t_end)
		// Create circles
		svg.selectAll("#protein")
			.data(pos)
			.enter()
			.append("circle")
			.attr("cx", function(d) {  return xScale(d.xloc);  })
			.attr("cy", function(d) {  return yScale(d.yloc);  })
			.attr("r", 2 )
			.attr("id","protein");
		
		// Set protein degradation times
		for (var a=0; a<nProteins; a++) {
			setTimeout(function(){
					pos.pop();
					svg.selectAll("#protein").data(pos).exit().remove();
				}, (t_end[a])*1000);
		}
		setInterval(function(){ update(pos); },0)
		setTimeout(function(){
			displayResults(t_end[nProteins-1]);
			clearInterval();
		}, t_end[nProteins-1]*1000);
	}

	function displayResults(t_end) {
		svg.selectAll("#protein").data([]).exit().remove();
	 
		// display t_end
		svg.selectAll("#result_text")
			.data([{x:0,y:0}])
			.enter()
			.append("text")
			.text(function(d) {return "Time Elapsed = " + t_end + " seconds"} )
			.attr("id","result_text")
			.attr("x", function(d) {return xScale(d.x)} )
			.attr("y", function(d) {return yScale(d.y)} )
			.style("text-anchor","middle")
			.style("fill","red")
			.style("font-size","15px");

		// Automatically run next trial after delay
		setTimeout(runSimulation, 2500);
	}


	//The data for our line
	lineData = []
	for (xx = 0; xx < 10; xx+=0.1) { 
	    lineData.push({ "x": xx, "y": 0.5*nProteins*lambda*Math.exp(-lambda*xx)});
	}

	//line accessor function
	var lineFunction = d3.svg.line()
	                      .x(function(d) { return xScale_hist(d.x); })
	                      .y(function(d) { return yScale_hist(d.y); })
	                     .interpolate("linear");


	//The line SVG Path we draw
	var lineGraph = svg.append("path")
	                            .attr("d", lineFunction(lineData))
	                            .attr("stroke", "red")
	                            .attr("stroke-width", 3)
	                            .attr("fill", "none");

	runSimulation();

})();

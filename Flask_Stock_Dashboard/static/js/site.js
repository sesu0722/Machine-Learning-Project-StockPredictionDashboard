var Site = function(){
	this.symbol = "";
};

Site.prototype.Init = function(){
	this.GetQuote();
	$("#symbol").on("click", function(){
		$(this).val("")
	});
};

Site.prototype.GetQuote = function(){
	// store the site context.
	var that = this;

	// pull the HTTP Request
	$.ajax({
		url: "/quote?symbol=" + that.symbol,
		method: "GET",
		cache: false
	}).done(function(data) {

		// set up a data context for just what we need.
		var context = {};
		context.shortName = data.shortName;
		context.symbol = data.symbol;
		context.price = data.ask;
      
		console.log(data.symbol)
		// call the request to load the chart and pass the data context with it.
		that.LoadChart(context);
	});

};

Site.prototype.SubmitForm = function(){
	this.symbol = $("#symbol").val();
	this.GetQuote();
	this.data()
}

Site.prototype.LoadChart = function(quote){

	var that = this;
	$.ajax({
		url: "/history?symbol=" + that.symbol,
		method: "GET",
		cache: false
	}).done(function(data) {
		that.RenderChart(JSON.parse(data), quote);
		console.log(quote)
	});
};

Site.prototype.data = function(symbol){
	var that = this;
	$.ajax({
		url: "/modeldata?symbol=" + that.symbol,
		method: "GET",
		cache: false
	}).done(function(data) {
        
		symbol=data.symbol
		console.log(symbol)
	})
}




Site.prototype.RenderChart = function(data, quote){
	var priceData = [];
	var dates = [];

	var title = quote.shortName  + " (" + quote.symbol + ") - " + numeral(quote.price).format('$0,0.00');

	for(var i in data.Close){
		var dt = i.slice(0,i.length-3);
		var dateString = moment.unix(dt).format("DD/MM/YY");
		var close = data.Close[i];
		if(close != null){
			priceData.push(data.Close[i]);
			dates.push(dateString);
		}
	}
	console.log(priceData)
	
	Highcharts.chart('chart_container', {
		title: {
			text: title
		},
		yAxis: {
			title: {
				text: ''
			}
		},
		xAxis: {
			categories :dates,
		},
		legend: {
			layout: 'vertical',
			align: 'right',
			verticalAlign: 'middle'
		},
		plotOptions: {
			series: {
				label: {
					connectorAllowed: false
				}
			},
			area: {
			}
		},
		series: [{
			type: 'area',
			color: '#6586bb',
			name: 'Price',
			data: priceData
		}],
		responsive: {
			rules: [{
				condition: {
					maxWidth: 640
				},
				chartOptions: {
					legend: {
						layout: 'horizontal',
						align: 'center',
						verticalAlign: 'bottom'
					}
				}
			}]
		}

	});

	
};



var site = new Site();

$(document).ready(()=>{
	site.Init();
})


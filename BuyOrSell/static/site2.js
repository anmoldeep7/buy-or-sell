var Site = function(){
	this.symbol = "ABNB";
};

Site.prototype.Init = function(){
	this.GetQuote();
	$("#symbol").on("click", function(){
		$(this).val("");
	});
};

Site.prototype.GetQuote = function(){
	// store the site context.
	var that = this;

	// pull the HTTP REquest
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

		if(data.quoteType="MUTUALFUND"){
			context.price = data.previousClose
		}
		var summary = {}
		summary.longBusinessSummary = data.longBusinessSummary;
		summary.symbol = data.symbol;
		summary.price = data.ask;
		summary.longName = data.longName;
		summary.logo_url = data.logo_url

		// call the request to load the chart and pass the data context with it.
		that.LoadChart(context);
		that.RenderQuote(summary);
	});
};

Site.prototype.SubmitForm = function(){
	this.symbol = $("#symbol").val();
	this.GetQuote();
}

Site.prototype.LoadChart = function(quote){

	var that = this;
	$.ajax({
		url: "/history?symbol=" + that.symbol,
		method: "GET",
		cache: false
	}).done(function(data) {
		that.RenderChart(JSON.parse(data), quote);
	});
};

Site.prototype.RenderChart = function(data, quote){
	var priceData = [];
	var dates = [];
	var volumeData = [];
	var title = " (" + quote.symbol + ")" 

	for(var i in data.Close){
		var dt = i.slice(0,i.length-3);
		var dateString = moment.unix(dt).format("DD/MM/YY");
		var close = data.Close[i];
		if(close != null){
			priceData.push(data.Close[i]);
            volumeData.push(data.Volume[i]);
			dates.push(dateString);
		}
	}

	Highcharts.chart('chart_container2', {
		title: {
			text: title+" Stock Volume traded ($)" 
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
            type: 'column',
            name: 'Volume',
            color: '#85bb65',
            data: volumeData
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

	Highcharts.chart('chart_container', {
		title: {
			text: title + " Stock Price ($)"
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
			color: '#85bb65',
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




Site.prototype.RenderQuote = function(summary){

	$("#sumTitle").html(summary.longName +" - "+ summary.symbol);
	$("#summary").html(summary.longBusinessSummary);
	var img =`<img src=${summary.logo_url} class="mw-100" alt="..."></img>`
	$("#sumImg").html(img);
	var quote= numeral(summary.price).format('$0,0.00');
	$("#sumSymbol").html("Actual Price");
	$("#sumQuote").html(quote);

	}


var site = new Site();

$(document).ready(()=>{
	site.Init();
})
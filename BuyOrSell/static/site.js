


var popularList = ['AAPL','AMC','GME'];
var url = '/quote?symbol='+popularList[0];
var url1 = '/quote?symbol='+popularList[1];
var url2 = '/quote?symbol='+popularList[2];

$.getJSON(url, function(data) {
    

    var text = `${data.shortName} - ${data.symbol}<br>
    Price: ${data.ask} $`
        
        $("#AAPL").html(text);
    });

$.getJSON(url1, function(data) {
    

    var text = `${data.shortName} - ${data.symbol}<br>
    Price: ${data.ask} $`
            
            $("#TESLA").html(text);
    });

$.getJSON(url2, function(data) {
    
    var text = `${data.shortName} - ${data.symbol}<br>
    Price: ${data.ask} $`
                
        $("#GME").html(text);
    });



$(document).ready(function(){
        $("#input").on("submit", function(){
          $("#pageloader").fadeIn();
    });//submit
});//document ready
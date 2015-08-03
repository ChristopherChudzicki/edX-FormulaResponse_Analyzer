$(document).ready( function(){
	$('.subm').addClass('hidden');
} );

$('.eval').click( function () {
	$(this).siblings('.subm').children('td').children('div').slideToggle(300);
	$(this).siblings('.subm').toggleClass('hidden');
} );

//add class "highlight" when hover over the row  
$('.eval').hover(function() {               
   $(this).addClass('highlight');  
}, function() {  
   $(this).removeClass('highlight');  
});  
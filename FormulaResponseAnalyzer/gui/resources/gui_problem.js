$(document).ready( function(){
	$('.details').hide();
} );

$('.summary').click( function () {
	$(this).next().slideToggle();
} );
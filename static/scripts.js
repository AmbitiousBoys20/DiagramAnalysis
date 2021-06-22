$(document).ready(function() {
    $('.tabs li').on('click', function() {
        // color active panel blue
        $('.tabs li').removeClass('is-active');
        $(this).addClass('is-active');

        var id = this.id;

        switch (id) {
            case 'Drawings':
                $('#diagramimg').removeClass('is-hidden');
                $('#comptab').addClass('is-hidden');
                $('#resultsimg').addClass('is-hidden');
                break;
            case 'Components':
                $('#comptab').removeClass('is-hidden');
                $('#diagramimg').addClass('is-hidden');
                $('#resultsimg').addClass('is-hidden');
                break;
            case 'Results':
                $('#resultsimg').removeClass('is-hidden');
                $('#diagramimg').addClass('is-hidden');
                $('#comptab').addClass('is-hidden');
                break;
        }
    });

    $('.panel-block').click(function() {
        // Show pdf of a diagram in the diagramimg block
        var filename = $(this).text();
        $('.panel-block').removeClass('is-active');
        $(this).addClass('is-active');
        var pdf = document.createElement("EMBED");
        pdf.src = filename;
        pdf.type = "application/pdf";
        pdf.width = "100%";
        pdf.height = "100%";
        pdf.style = "min-height: 85vh;"
        $('#diagramimg').empty();
        $('#diagramimg').append(pdf);
    });

    $('#matchbtn').click(async function() {
        if ($(this).is("[disabled=disabled]")) {
            alert('Wait for the Template Matcher to finish!');
            return;
        }

        $(this).text("");
        $(this).append('<div class="loader"></div>');
        $(this).attr("disabled", true);
        var data = {
            'filename': $('.panel-block.is-active').text(),
            'templatefolder': $('#templatefolder').val(),
            'threshold': $('#threshold').val(),
            'scalemin': $('#scalemin').val(),
            'scalemax': $('#scalemax').val(),
            'scalenum': $('#scalenum').val()
        };
        await $.ajax({
            type: 'POST',
            url: '/matchtemplates',
            data: data,
            success: function(result) {
                names = result['names'];
                counts = result['counts'];
                loc = result['location'];
                labels = result['labels'];
                components = result['components'];
                $('#componenttable').empty();
                $('#labeltable').empty();

                $('#componenttable').append('<tr><th>Component</th><th>Counts</th></tr>');
                $('#labeltable').append('<tr><th>Component</th><th>Location</th><th>Label</th></tr>');

                // make a table for the count
                for (var i = 0; i < names.length; i++) {
                    $('#componenttable').append(
                        '<tr><td>'+ names[i] + '</td><td>' + counts[i] + '</td></tr>'
                    );
                }
                
                // make the table for the labels and components
                for (var i = 0; i < loc.length; i++) {
                    $('#labeltable').append(
                        '<tr><td>'+ components[i] + '</td><td>' + loc[i] +
                         '</td><td>' + labels[i] + '</td</tr>'
                    );
                }
                
                var pdf = document.createElement("EMBED");
                pdf.src = 'static/diagrams/out.pdf';
                pdf.type = "application/pdf";
                pdf.width = "100%";
                pdf.height = "100%";
                pdf.style = "min-height: 85vh;"
                $('#resultsimg').empty();
                $('#resultsimg').append(pdf);

                // go to results tab
                $('#Results').addClass('is-active');
                $('#Drawings').removeClass('is-active');
                $('#Components').removeClass('is-active');
                $('#resultsimg').removeClass('is-hidden');
                $('#comptab').addClass('is-hidden');
                $('#diagramimg').addClass('is-hidden');
            },
            statusCode: {
                400: function() {
                    alert('Diagram or folder not found!');
                    unlockbtn();
                }
            }
        });
        unlockbtn();
    });
});

function unlockbtn() {
    $('#matchbtn').attr('disabled', false);
    $(".loader").remove();
    $('#matchbtn').text("Match Templates");
}
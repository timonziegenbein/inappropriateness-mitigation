$('input[type="range"]').rangeslider({
  // Feature detection the default is `true`.
    // Set this to `false` if you want to use
    // the polyfill also in Browsers which support
    // the native <input type="range"> element.
    polyfill: false,

    // Default CSS classes
    rangeClass: 'rangeslider',
    disabledClass: 'rangeslider--disabled',
    horizontalClass: 'rangeslider--horizontal',
    fillClass: 'rangeslider__fill',
    handleClass: 'rangeslider__handle',

    // Callback function
    onInit: function() {
      $rangeEl = this.$range;
      // add value label to handle
      var $handle = $rangeEl.find('.rangeslider__handle');

      
      // get range index labels 
      var rangeLabels = this.$element.attr('labels');
      rangeLabels = rangeLabels.split(', ');
      
      // add labels
      $rangeEl.append('<div class="rangeslider__labels"></div>');
      $(rangeLabels).each(function(index, value) {
        $rangeEl.find('.rangeslider__labels').append('<span class="rangeslider__labels__label">' + value + '</span>');
      })
    },

    // Callback function
    onSlide: function(position, value) {
      var $handle = this.$range.find('.rangeslider__handle__value');
      $handle.text(this.value);
    },

    // Callback function
    onSlideEnd: function(position, value) {}
});

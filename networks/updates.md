1. network updates
    1. change the filter size of first conv layer in each 'build block';
    2. change the order of conv and pool in shortcut when there is a feature-increase;
    3. when the feature size is downsampled to 1, stop downsampling;
    4. when feature size is less than filter size, set filter size to be feature size;
2. add visualize notebook
3. modified the accuracy test after train process
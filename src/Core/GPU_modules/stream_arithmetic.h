struct StreamInfo {
    const unsigned long long stream_size;
    const int slices_per_stream;
};

const StreamInfo calculate_stream_size(int no_of_streams, int N, int M, int Z) {

    const unsigned long long n = (unsigned long long)N*(unsigned long long)M*(unsigned long long)Z;
    // the absolute minimum number of pixels that should be processed per stream
    const unsigned long long ideal_stream_size = n / no_of_streams;

    // find the integer multiple of slices of the data that each stream needs
    // to process in order to cover AT LEAST ideal_stream_size, but will in
    // practice be a bit larger
    float ideal_slices_per_stream;
    int slices_per_stream;

    // calculate the size of a stream
    unsigned long long stream_size;

    if(Z == 1) {
        // 2D case
        ideal_slices_per_stream = (float)ideal_stream_size/(float)N;
        slices_per_stream = ceil(ideal_slices_per_stream);
        stream_size = slices_per_stream * (unsigned long long)N;
    }
    else {
        // 3D case
        ideal_slices_per_stream = (float)ideal_stream_size/(float)(N*M);
        slices_per_stream = ceil(ideal_slices_per_stream);
        stream_size = slices_per_stream * (unsigned long long)N * (unsigned long long)M;
    }

    const struct StreamInfo stream_info = { stream_size, slices_per_stream };

    return stream_info;
}

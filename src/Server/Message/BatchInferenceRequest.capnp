@0xb202a66272e75add;

struct BatchInferenceRequest {

    apiKey    @0 :Text;
    modelId   @1 :UInt32;
    batchSize @2 :UInt32;
    inputs    @3 :List(Input);
    
    struct Input {
        enum Type {
            bool    @0;
            int8    @1;
            int16   @2;
            int32   @3;
            int64   @4;
            uint8   @5;
            uint16  @6;
            uint32  @7;
            uint64  @8;
            fp32    @9;
            fp64    @10;
            string  @11;
            binary  @12;
        }

        name    @0  :Text;
        type    @1  :Type;
        value   @2  :AnyPointer;
    }
}


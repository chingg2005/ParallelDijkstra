#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
constexpr int INF = INT_MAX / 4;
struct Options { int n=5, source=0, seed=12345; double density=0.02; string mode="small", csv="results/benchmark.csv"; };
int start_of(int r,int n,int p){return r*(n/p)+min(r,n%p);}
int count_of(int r,int n,int p){return n/p+(r<n%p);}
Options parse(int argc,char**argv){
    Options o;
    if(argc>1) o.mode=argv[1];
    if(o.mode=="large") o.n=2000;
    else if(o.mode!="small") throw runtime_error("Mode must be small or large");
    for(int i=2;i<argc;i++){
        string a=argv[i];
        if((a=="--n"||a=="--source"||a=="--seed"||a=="--density"||a=="--csv")&&i+1>=argc) throw runtime_error("Missing value for "+a);
        if(a=="--n") o.n=stoi(argv[++i]);
        else if(a=="--source") o.source=stoi(argv[++i]);
        else if(a=="--seed") o.seed=stoi(argv[++i]);
        else if(a=="--density") o.density=stod(argv[++i]);
        else if(a=="--csv") o.csv=argv[++i];
        else throw runtime_error("Unknown argument: "+a);
    }
    if(o.n<1||o.n>30000||o.source<0||o.source>=o.n||o.density<0||o.density>1) throw runtime_error("Invalid arguments");
    if(o.mode=="small"&&o.n!=5) throw runtime_error("small mode requires n=5");
    if(1LL*o.n*o.n>INT_MAX) throw runtime_error("Graph too large for MPI count");
    return o;
}
vector<int> generate(const Options&o){
    int n=o.n; vector<int> g((size_t)n*n,INF);
    for(int i=0;i<n;i++) g[(size_t)i*n+i]=0;
    auto edge=[&](int u,int v,int w){g[(size_t)u*n+v]=w;};
    if(o.mode=="small"){
        edge(0,1,10);edge(0,2,3);edge(2,1,1);edge(1,3,2);
        edge(2,3,8);edge(2,4,2);edge(4,3,7);
    }else{
        mt19937 rng(o.seed);
        bernoulli_distribution exists(o.density);
        uniform_int_distribution<int> weight(1,20);
        for(int u=0;u<n;u++) for(int v=0;v<n;v++)
            if(u!=v&&exists(rng)) edge(u,v,weight(rng));
        for(int u=0;u<n-1;u++) edge(u,u+1,min(g[(size_t)u*n+u+1],1));
    }
    return g;
}
vector<int> sequential(const vector<int>&g,int n,int source){
    vector<int>d(n,INF),used(n,0);d[source]=0;
    for(int it=0;it<n;it++){
        int u=-1;
        for(int v=0;v<n;v++) if(!used[v]&&(u<0||d[v]<d[u]))u=v;
        if(u<0||d[u]==INF)break;
        used[u]=1;
        for(int v=0;v<n;v++){
            int w=g[(size_t)u*n+v];
            if(!used[v]&&w!=INF&&d[u]<=INF-w)
                d[v]=min(d[v],d[u]+w);
        }
    }
    return d;
}
vector<int> parallel(const vector<int>&g,int n,int source,int rank,int p){
    int begin=start_of(rank,n,p),cnt=count_of(rank,n,p);
    vector<int> local(cnt,INF), used(cnt,0);
    if(source>=begin&&source<begin+cnt)local[source-begin]=0;
    for(int it=0;it<n;it++){
        // MPI_2INT + MPI_MINLOC: pair(distance, vertex ID).
        int candidate[2]={INF,INT_MAX};
        for(int j=0;j<cnt;j++) if(!used[j]){
            int v=begin+j;
            if(local[j]<candidate[0]||(local[j]==candidate[0]&&v<candidate[1])){
                candidate[0]=local[j];candidate[1]=v;
            }
        }
        int best[2]={INF,INT_MAX};
        MPI_Allreduce(candidate,best,1,MPI_2INT,MPI_MINLOC,MPI_COMM_WORLD);
        if(best[0]==INF)break;
        int u=best[1], du=best[0];
        if(u>=begin&&u<begin+cnt)used[u-begin]=1;
        for(int j=0;j<cnt;j++){
            if(used[j])continue;
            int v=begin+j,w=g[(size_t)u*n+v];
            if(w!=INF&&du<=INF-w)local[j]=min(local[j],du+w);
        }
    }
    vector<int> result;
    if(rank==0)result.resize(n);
    vector<int> counts(p),displs(p);
    for(int r=0;r<p;r++){counts[r]=count_of(r,n,p);displs[r]=start_of(r,n,p);}
    MPI_Gatherv(local.data(),cnt,MPI_INT,rank==0?result.data():nullptr,
                counts.data(),displs.data(),MPI_INT,0,MPI_COMM_WORLD);
    return result;
}
int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank=0,p=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&p);
    try{
        Options o=parse(argc,argv);
        vector<int> g;
        if(rank==0)g=generate(o);
        else g.resize((size_t)o.n*o.n);
        MPI_Bcast(g.data(),o.n*o.n,MPI_INT,0,MPI_COMM_WORLD);
        vector<int> baseline;double seq_time=0;
        if(rank==0){
            auto t0=chrono::steady_clock::now();
            baseline=sequential(g,o.n,o.source);
            seq_time=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t0=MPI_Wtime();
        vector<int> answer=parallel(g,o.n,o.source,rank,p);
        double local_time=MPI_Wtime()-t0,mpi_time=0;
        MPI_Reduce(&local_time,&mpi_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if(rank==0){
            bool valid=answer==baseline;
            cout<<fixed<<setprecision(6);
            cout<<"Mode: "<<o.mode<<" | vertices: "<<o.n<<" | processes: "<<p<<"\n";
            cout<<"Sequential time: "<<seq_time<<" s\n";
            cout<<"MPI time: "<<mpi_time<<" s\n";
            cout<<"Speedup: "<<(mpi_time>0?seq_time/mpi_time:0)<<"\n";
            cout<<"Efficiency: "<<(mpi_time>0?100*seq_time/(mpi_time*p):0)<<"%\n";
            cout<<"Verification: "<<(valid?"PASS":"FAIL")<<"\n";
            int shown=min(o.n,o.mode=="small"?o.n:20);
            for(int v=0;v<shown;v++)cout<<o.source<<" -> "<<v<<" = "<<(answer[v]==INF?"INF":to_string(answer[v]))<<"\n";
            if(o.n>shown)cout<<"... first "<<shown<<" vertices shown\n";
            if(!o.csv.empty()){
                std::filesystem::create_directories("results");
                ifstream check(o.csv);bool header=!check.good()||check.peek()==ifstream::traits_type::eof();check.close();
                ofstream out(o.csv,ios::app);
                if(!out)cerr<<"Warning: cannot write CSV: "<<o.csv<<" (create directory first)\n";
                else{
                    if(header)out<<"mode,n,processes,seed,density,sequential_s,mpi_s,speedup,efficiency_percent,verified\n";
                    out<<o.mode<<','<<o.n<<','<<p<<','<<o.seed<<','<<o.density<<','<<setprecision(9)
                       <<seq_time<<','<<mpi_time<<','<<(mpi_time>0?seq_time/mpi_time:0)<<','
                       <<(mpi_time>0?100*seq_time/(mpi_time*p):0)<<','<<(valid?1:0)<<'\n';
                }
            }
        }
    }catch(const exception&e){
        cerr<<"Rank "<<rank<<" error: "<<e.what()<<"\n";
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();return 0;
}

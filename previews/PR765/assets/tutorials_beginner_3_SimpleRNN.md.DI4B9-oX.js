import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.B66_d8zn.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR765/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR765/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR765/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR765/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56155</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51634</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48130</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44775</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41730</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39970</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37805</span></span>
<span class="line"><span>Validation: Loss 0.36168 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36575 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37147</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35029</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32766</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31354</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30314</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28670</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28263</span></span>
<span class="line"><span>Validation: Loss 0.25331 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25601 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25992</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23251</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23159</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21970</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21413</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20122</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19108</span></span>
<span class="line"><span>Validation: Loss 0.17725 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17863 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17746</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17003</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16403</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15333</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14899</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14076</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13203</span></span>
<span class="line"><span>Validation: Loss 0.12650 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12732 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12824</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12125</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11687</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11114</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10629</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10114</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09609</span></span>
<span class="line"><span>Validation: Loss 0.09159 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09236 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09222</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08783</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08687</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08263</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07747</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07295</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07054</span></span>
<span class="line"><span>Validation: Loss 0.06714 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06797 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06852</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06586</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06274</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05935</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05869</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05518</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05282</span></span>
<span class="line"><span>Validation: Loss 0.04980 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05064 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05199</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05052</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04683</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04563</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04327</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04013</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03656</span></span>
<span class="line"><span>Validation: Loss 0.03732 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03811 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03692</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03817</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03616</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03418</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03262</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03246</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02979</span></span>
<span class="line"><span>Validation: Loss 0.02841 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02917 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02727</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02868</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02784</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02691</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02628</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02581</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02394</span></span>
<span class="line"><span>Validation: Loss 0.02231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02298 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02201</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02245</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02181</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02377</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01999</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02010</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01900</span></span>
<span class="line"><span>Validation: Loss 0.01814 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01873 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01924</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01807</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01854</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01697</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01681</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01583</span></span>
<span class="line"><span>Validation: Loss 0.01525 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01575 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01717</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01533</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01477</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01587</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01358</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01288</span></span>
<span class="line"><span>Validation: Loss 0.01318 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01362 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01432</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01403</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01281</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01240</span></span>
<span class="line"><span>Validation: Loss 0.01163 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01203 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01252</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01205</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01102</span></span>
<span class="line"><span>Validation: Loss 0.01042 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01078 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01135</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01098</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01085</span></span>
<span class="line"><span>Validation: Loss 0.00944 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01050</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00912</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00912</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00941</span></span>
<span class="line"><span>Validation: Loss 0.00862 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00892 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00964</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00889</span></span>
<span class="line"><span>Validation: Loss 0.00792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00819 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00732</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00845</span></span>
<span class="line"><span>Validation: Loss 0.00732 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00758 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00817</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00743</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00720</span></span>
<span class="line"><span>Validation: Loss 0.00679 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00703 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00701</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00620</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00687</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00705</span></span>
<span class="line"><span>Validation: Loss 0.00633 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00656 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00664</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00701</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00663</span></span>
<span class="line"><span>Validation: Loss 0.00592 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00613 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00638</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00605</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00619</span></span>
<span class="line"><span>Validation: Loss 0.00555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00575 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00600</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00568</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00553</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00536</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00502</span></span>
<span class="line"><span>Validation: Loss 0.00523 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00541 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00525</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00574</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00514</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00514</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00506</span></span>
<span class="line"><span>Validation: Loss 0.00493 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00511 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56221</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51193</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47806</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44963</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41400</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40447</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38935</span></span>
<span class="line"><span>Validation: Loss 0.37374 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36000 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37391</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35572</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33278</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30928</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29693</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28708</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.25927</span></span>
<span class="line"><span>Validation: Loss 0.26116 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25329 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25743</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24862</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23203</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21817</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20624</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19716</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19275</span></span>
<span class="line"><span>Validation: Loss 0.18169 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17738 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17692</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17103</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16486</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15185</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14460</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14375</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13483</span></span>
<span class="line"><span>Validation: Loss 0.12918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12645 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12948</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12086</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11554</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11131</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10469</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10179</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09818</span></span>
<span class="line"><span>Validation: Loss 0.09400 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09163 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09307</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08849</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08527</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08320</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07620</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07439</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06669</span></span>
<span class="line"><span>Validation: Loss 0.06963 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06719 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06717</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06504</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06307</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06213</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05793</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05462</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05308</span></span>
<span class="line"><span>Validation: Loss 0.05233 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04971 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05096</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04903</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04690</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04357</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04350</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04252</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04204</span></span>
<span class="line"><span>Validation: Loss 0.03969 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03712 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03901</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03490</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03552</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03441</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03423</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03170</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02992</span></span>
<span class="line"><span>Validation: Loss 0.03056 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02817 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02946</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02663</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02858</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02602</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02593</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02537</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02336</span></span>
<span class="line"><span>Validation: Loss 0.02419 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02205 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02287</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02310</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02091</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01990</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01950</span></span>
<span class="line"><span>Validation: Loss 0.01976 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01790 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01931</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01848</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01866</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01761</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01660</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01540</span></span>
<span class="line"><span>Validation: Loss 0.01663 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01503 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01590</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01580</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01531</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01482</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01440</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01526</span></span>
<span class="line"><span>Validation: Loss 0.01438 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01297 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01449</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01235</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01280</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01295</span></span>
<span class="line"><span>Validation: Loss 0.01268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01144 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01188</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01119</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01164</span></span>
<span class="line"><span>Validation: Loss 0.01135 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01023 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01056</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01101</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01074</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00992</span></span>
<span class="line"><span>Validation: Loss 0.01028 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00926 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00939</span></span>
<span class="line"><span>Validation: Loss 0.00939 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00844 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00737</span></span>
<span class="line"><span>Validation: Loss 0.00864 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00776 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00589</span></span>
<span class="line"><span>Validation: Loss 0.00800 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00717 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00700</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00736</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00607</span></span>
<span class="line"><span>Validation: Loss 0.00743 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00666 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00702</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00678</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00725</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00697</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00652</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00631</span></span>
<span class="line"><span>Validation: Loss 0.00694 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00620 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00660</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00673</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00680</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00637</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00548</span></span>
<span class="line"><span>Validation: Loss 0.00650 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00581 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00626</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00654</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00557</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00568</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00640</span></span>
<span class="line"><span>Validation: Loss 0.00611 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00545 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00577</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00552</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00603</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00502</span></span>
<span class="line"><span>Validation: Loss 0.00575 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00513 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00503</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00517</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00528</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00524</span></span>
<span class="line"><span>Validation: Loss 0.00543 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00484 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxDeviceUtils)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.4</span></span>
<span class="line"><span>Commit 48d4fd48430 (2024-06-04 10:41 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 4 default, 0 interactive, 2 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_AMDGPU_LOGGING_ENABLED = true</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 4</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.1+1</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
